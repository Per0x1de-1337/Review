package github

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"time"

	// "github.com/keploy/keploy-review-agent/internal/analyzer/llm"
	// "github.com/keploy/keploy-review-agent/internal/analyzer/static"
	"github.com/keploy/keploy-review-agent/pkg/models"
)

var pullnumber int

func PullRequestNumber(currentpullnumber int) int {
	pullnumber = currentpullnumber
	return pullnumber
}

var comment string

// Client handles communication with the GitHub API
type Client struct {
	token      string
	httpClient *http.Client
	baseURL    string
}

// NewClient creates a new GitHub API client
func NewClient(token string) *Client {
	return &Client{
		token: token,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		baseURL: "https://api.github.com",
	}
}

// GetChangedFiles fetches the files changed in a pull request
func (c *Client) GetChangedFiles(ctx context.Context, owner, repo string, pullnumber int) ([]*models.File, error) {
	url := fmt.Sprintf("%s/repos/%s/%s/pulls/%d/files", c.baseURL, owner, repo, pullnumber)
	fmt.Printf("url: %s\n", url)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "token "+c.token)
	req.Header.Set("Accept", "application/vnd.github.v3+json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("GitHub API error: %s, %s", resp.Status, string(body))
	}

	var prFiles []struct {
		Filename string `json:"filename"`
		Status   string `json:"status"`
		RawURL   string `json:"raw_url"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&prFiles); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var files []*models.File

	// Fetch content for each file
	for _, prFile := range prFiles {
		// Skip deleted files
		if prFile.Status == "removed" {
			continue
		}

		// Fetch file content
		content, err := c.getFileContent(ctx, owner, repo, prFile.Filename)
		if err != nil {
			return nil, fmt.Errorf("failed to get content for %s: %w", prFile.Filename, err)
		}

		files = append(files, &models.File{
			Path:    prFile.Filename,
			Content: content,
		})
	}

	return files, nil
}

// getFileContent fetches the content of a specific file
	func (c *Client) getFileContent(ctx context.Context, owner, repo, path string) (string, error) {
		url := fmt.Sprintf("%s/repos/%s/%s/contents/%s", c.baseURL, owner, repo, path)
		log.Printf("url: %s\n", url)
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return "", fmt.Errorf("failed to create request: %w", err)
		}

		req.Header.Set("Authorization", "token "+c.token)
		req.Header.Set("Accept", "application/vnd.github.v3+json")

		resp, err := c.httpClient.Do(req)
		if err != nil {
			return "", fmt.Errorf("failed to send request: %w", err)
		}
		defer resp.Body.Close()
		log.Printf("resp: %s\n", resp)
		if resp.StatusCode != http.StatusOK {
			body, _ := ioutil.ReadAll(resp.Body)
			return "", fmt.Errorf("GitHub API error: %s, %s", resp.Status, string(body))
		}

		var fileData struct {
			Content  string `json:"content"`
			Encoding string `json:"encoding"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&fileData); err != nil {
			return "", fmt.Errorf("failed to decode response: %w", err)
		}

		// GitHub returns base64 encoded content
		if fileData.Encoding == "base64" {
			decoded, err := base64Decode(fileData.Content)
			if err != nil {
				return "", fmt.Errorf("failed to decode base64 content: %w", err)
			}
			return string(decoded), nil
		}

		return fileData.Content, nil
	}

// CreateReview creates a pull request review with comments
func (c *Client) CreateReview(ctx context.Context, owner, repo string, pullnumber int, comments []*models.ReviewComment) error {
		url := fmt.Sprintf("%s/repos/%s/%s/issues/%d/comments", c.baseURL, owner, repo, pullnumber)
	
		// 📝 **Format comments as Markdown**
	var markdownComment string
	markdownComment += "### 📝 Automated Review Comments\n\n"
	markdownComment += "Thank you for raising this pull request. Below are the review comments:\n\n"

	for _, comment := range comments {
		markdownComment += fmt.Sprintf(
			"- **File:** `%s`\n  - **Position:** %d\n  - **Comment:** %s\n\n",
			comment.Path, comment.Position, comment.Body,
		)
	}

	log.Printf("Generated Markdown Comment:\n%s", markdownComment)

	// 📨 **Prepare request payload**
	payload := map[string]interface{}{
		"body": markdownComment,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal review payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "token "+c.token)
	req.Header.Set("Accept", "application/vnd.github.v3+json")
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := ioutil.ReadAll(resp.Body)
		log.Printf("GitHub API Response: %s", string(body))
		return fmt.Errorf("GitHub API error: %s, %s", resp.Status, string(body))
	}

	log.Println("Review successfully posted to GitHub.")
	return nil
	}

// base64Decode decodes base64 content (helper function)
func base64Decode(content string) ([]byte, error) {
	decoded, err := base64.StdEncoding.DecodeString(content)
	if err != nil {
		return nil, fmt.Errorf("base64 decode error: %w", err)
	}
	return decoded, nil
}

// ProcessPullRequestReview integrates GetChangedFiles and CreateReview
func (c *Client) ProcessPullRequestReview(ctx context.Context, owner, repo string, pullnumber int) error {
	// Fetch the changed files in the PR
	changedFiles, err := c.GetChangedFiles(ctx, owner, repo, pullnumber)
	if err != nil {
		return fmt.Errorf("failed to get changed files: %w", err)
	}
	fmt.Println("changedFiles: track01 ", changedFiles)
	// Generate comments for the review
	var reviewComments []*models.ReviewComment
	for _, file := range changedFiles {
		// For simplicity, let's assume we're generating a comment for each file
		reviewComments = append(reviewComments, &models.ReviewComment{
			Path:     file.Path,
			Position: 1, // Assuming position 1 for the sake of this example
			Body:     "Please review this file.",
		})
	}

	// Create the review
	err = c.CreateReview(ctx, owner, repo, pullnumber, reviewComments)	
	if err != nil {
		return fmt.Errorf("failed to create review: %w", err)
	}

	return nil
}
