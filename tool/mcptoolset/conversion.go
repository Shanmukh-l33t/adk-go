// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mcptoolset

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/tool"
)

// ToMCPTool converts t into an [mcp.Tool] definition suitable for registration
// on an [mcp.Server] with [mcp.Server.AddTool].
//
// If t implements Declaration() *genai.FunctionDeclaration, ToMCPTool sets
// InputSchema from ParametersJsonSchema when present, or converts Parameters
// into a JSON Schema map. When t has no declaration or declares neither
// parameter field, InputSchema defaults to {"type": "object"} so
// [mcp.Server.AddTool] accepts the tool.
//
//	mcpTool, err := mcptoolset.ToMCPTool(adkTool)
//	if err != nil {
//		return err
//	}
//	server.AddTool(mcpTool, handler)
func ToMCPTool(t tool.Tool) (*mcp.Tool, error) {
	if t == nil {
		return nil, errors.New("mcptoolset: tool is nil")
	}

	// Diverges from adk-python's adk_to_mcp_tool_type, which returns {} when a
	// tool has no declaration or parameters: go-sdk's Server.AddTool panics
	// unless InputSchema has type "object".
	inputSchema := any(map[string]any{"type": "object"})
	if d, ok := t.(interface {
		Declaration() *genai.FunctionDeclaration
	}); ok {
		if decl := d.Declaration(); decl != nil {
			switch {
			case decl.ParametersJsonSchema != nil:
				inputSchema = decl.ParametersJsonSchema
			case decl.Parameters != nil:
				schema, err := geminiToJSONSchema(decl.Parameters)
				if err != nil {
					return nil, fmt.Errorf("mcptoolset: convert parameters schema for tool %q: %w", t.Name(), err)
				}
				inputSchema = schema
			}
		}
	}

	return &mcp.Tool{
		Name:        t.Name(),
		Description: t.Description(),
		InputSchema: inputSchema,
	}, nil
}

// geminiToJSONSchema converts a Gemini Schema into a JSON Schema map.
func geminiToJSONSchema(schema *genai.Schema) (map[string]any, error) {
	if schema == nil {
		return nil, errors.New("mcptoolset: gemini schema is nil")
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("mcptoolset: marshal gemini schema: %w", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("mcptoolset: unmarshal gemini schema: %w", err)
	}
	normalizeGeminiSchemaMap(raw)
	return raw, nil
}

// normalizeGeminiSchemaMap normalizes a marshaled genai.Schema map in place into
// standard JSON Schema, recursively normalizing schema child nodes.
func normalizeGeminiSchemaMap(raw map[string]any) {
	delete(raw, "propertyOrdering")

	typeStr, _ := raw["type"].(string)
	geminiType := genai.Type(strings.ToUpper(typeStr))

	// Omitted/unspecified types impose no JSON Schema type constraint. Vertex
	// function declarations also mark a union as OBJECT plus anyOf; that OBJECT
	// is a placeholder, so the anyOf branches carry the type.
	isUnionPlaceholder := false
	if geminiType == genai.TypeObject {
		if anyOf, ok := raw["anyOf"].([]any); ok {
			for _, elem := range anyOf {
				if subSchema, ok := elem.(map[string]any); ok {
					subTypeStr, _ := subSchema["type"].(string)
					subType := genai.Type(strings.ToUpper(subTypeStr))
					if subType != "" && subType != genai.TypeUnspecified && subType != genai.TypeObject {
						isUnionPlaceholder = true
						break
					}
				}
			}
		}
	}

	if geminiType == "" || geminiType == genai.TypeUnspecified || isUnionPlaceholder {
		delete(raw, "type")
	} else {
		raw["type"] = strings.ToLower(string(geminiType))
	}

	if nullable, ok := raw["nullable"].(bool); ok && !nullable {
		delete(raw, "nullable")
	}

	if geminiType != genai.TypeString {
		delete(raw, "pattern")
		delete(raw, "minLength")
		delete(raw, "maxLength")
	}

	if geminiType != genai.TypeNumber && geminiType != genai.TypeInteger {
		delete(raw, "minimum")
		delete(raw, "maximum")
	}

	if geminiType != genai.TypeArray {
		delete(raw, "items")
		delete(raw, "minItems")
		delete(raw, "maxItems")
	} else if items, ok := raw["items"].(map[string]any); ok {
		normalizeGeminiSchemaMap(items)
	}

	if geminiType != genai.TypeObject {
		delete(raw, "properties")
		delete(raw, "required")
		delete(raw, "minProperties")
		delete(raw, "maxProperties")
	} else if props, ok := raw["properties"].(map[string]any); ok {
		for _, v := range props {
			if propSchema, ok := v.(map[string]any); ok {
				normalizeGeminiSchemaMap(propSchema)
			}
		}
	}

	if anyOf, ok := raw["anyOf"].([]any); ok {
		for _, elem := range anyOf {
			if subSchema, ok := elem.(map[string]any); ok {
				normalizeGeminiSchemaMap(subSchema)
			}
		}
	}
}
