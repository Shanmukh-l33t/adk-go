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

package vertexairag

import (
	"context"
	"errors"
	"fmt"
	"io"

	aiplatform "cloud.google.com/go/aiplatform/apiv1beta1"
	"cloud.google.com/go/aiplatform/apiv1beta1/aiplatformpb"
	aiplatformrest "google.golang.org/api/aiplatform/v1beta1"
	"google.golang.org/api/option"

	aiplatformutil "google.golang.org/adk/v2/util/aiplatform"
)

type ragBackend interface {
	uploadFile(ctx context.Context, corpusName, displayName string, content io.Reader) error
	listFiles(ctx context.Context, req *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error)
	retrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error)
}

type vertexRAGClient struct {
	ragClient     *aiplatform.VertexRagClient
	ragDataClient *aiplatform.VertexRagDataClient
	restService   *aiplatformrest.Service
}

func newVertexRAGClient(ctx context.Context, location string, opts ...option.ClientOption) (*vertexRAGClient, error) {
	ctx = context.WithoutCancel(ctx)
	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("client option cannot be nil")
		}
	}

	grpcOpts := append([]option.ClientOption{
		option.WithEndpoint(aiplatformutil.HostPortURL(location)),
	}, opts...)
	restOpts := append([]option.ClientOption{
		option.WithEndpoint("https://" + aiplatformutil.HostURL(location)),
	}, opts...)

	ragClient, err := aiplatform.NewVertexRagClient(ctx, grpcOpts...)
	if err != nil {
		return nil, fmt.Errorf("create VertexRagClient: %w", err)
	}
	ragDataClient, err := aiplatform.NewVertexRagDataClient(ctx, grpcOpts...)
	if err != nil {
		_ = ragClient.Close()
		return nil, fmt.Errorf("create VertexRagDataClient: %w", err)
	}
	restService, err := aiplatformrest.NewService(ctx, restOpts...)
	if err != nil {
		_ = ragClient.Close()
		_ = ragDataClient.Close()
		return nil, fmt.Errorf("create aiplatform REST service: %w", err)
	}

	return &vertexRAGClient{
		ragClient:     ragClient,
		ragDataClient: ragDataClient,
		restService:   restService,
	}, nil
}

func (c *vertexRAGClient) uploadFile(ctx context.Context, corpusName, displayName string, content io.Reader) error {
	req := &aiplatformrest.GoogleCloudAiplatformV1beta1UploadRagFileRequest{
		RagFile: &aiplatformrest.GoogleCloudAiplatformV1beta1RagFile{
			DisplayName: displayName,
		},
		UploadRagFileConfig: &aiplatformrest.GoogleCloudAiplatformV1beta1UploadRagFileConfig{},
	}
	_, err := c.restService.Media.Upload(corpusName, req).Media(content).Context(ctx).Do()
	return err
}

func (c *vertexRAGClient) listFiles(ctx context.Context, req *aiplatformpb.ListRagFilesRequest) ([]*aiplatformpb.RagFile, string, error) {
	it := c.ragDataClient.ListRagFiles(ctx, req)
	return it.InternalFetch(int(req.GetPageSize()), req.GetPageToken())
}

func (c *vertexRAGClient) retrieveContexts(ctx context.Context, req *aiplatformpb.RetrieveContextsRequest) (*aiplatformpb.RetrieveContextsResponse, error) {
	return c.ragClient.RetrieveContexts(ctx, req)
}
