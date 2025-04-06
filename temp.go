// package llm

// import (
// 	"bytes"
// 	"context"
// 	"encoding/json"
// 	"fmt"
// 	"io"
// 	"log"
// 	"net/http"
// 	"path/filepath"
// 	"regexp"
// 	"strings"
// 	"time"

// 	"github.com/keploy/keploy-review-agent/pkg/models"
// )

// const (
// 	googleAIEndpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
// 	maxRetries       = 3
// 	baseDelay        = 1 * time.Second
// )

// var jsonRegex = regexp.MustCompile(`(?s)\[\s*{.*?}\s*\]`)

// type GoogleAIClient struct {
// 	apiKey     string
// 	httpClient *http.Client
// 	config     *AIConfig
// }

// type AIConfig struct {
// 	MaxTokens   int
// 	Temperature float64
// 	MinSeverity models.Severity
// }

// func NewGoogleAIClient(apiKey string, cfg *AIConfig) *GoogleAIClient {

// 	return &GoogleAIClient{
// 		apiKey: apiKey,
// 		httpClient: &http.Client{
// 			Timeout: 30 * time.Second,
// 		},
// 		config: cfg,
// 	}
// }

// func (g *GoogleAIClient) AnalyzeCode(ctx context.Context, files []*models.File) ([]*models.Issue, error) {

// 	var allIssues []*models.Issue

// 	for _, file := range files {

// 		if shouldSkipFile(file.Path) {
// 			fmt.Println("Skipping file:", file.Path)
// 			continue
// 		}
	
// 		issues, err := g.analyzeFile(ctx, file)
// 		if err != nil {
// 			log.Printf("AI analysis failed for %s: %v", file.Path, err)
// 			continue
// 		}

// 		allIssues = append(allIssues, filterIssues(issues, g.config.MinSeverity)...)
// 		fmt.Println("Filtered issues for", file.Path, ":", allIssues)
// 	}

// 	fmt.Println("AnalyzeCode: Completed analysis with", len(allIssues), "total issues")
// 	return allIssues, nil
// }

// func shouldSkipFile(path string) bool {
// 	ext := filepath.Ext(path)
// 	skip := !(ext == ".go" || ext == ".js" || ext == ".ts" || ext == ".py")

// 	return skip
// }

// func (g *GoogleAIClient) analyzeFile(ctx context.Context, file *models.File) ([]*models.Issue, error) {
// 	fmt.Println("Analyzing file:", file.Path)
// 	prompt := buildPrompt(file.Content)


// 	var response string
// 	var err error

// 	for i := 0; i < maxRetries; i++ {
// 		fmt.Println("Attempt", i+1, "to call generateContent")
// 		response, err = g.generateContent(ctx, prompt)
// 		if err == nil {
// 			break
// 		}
// 		fmt.Println("Retrying in", baseDelay*time.Duration(i*i), "due to error:", err)
// 		time.Sleep(baseDelay * time.Duration(i*i))
// 	}
// 	if err != nil {
// 		fmt.Println("Failed to analyze file after retries:", err)
// 		return nil, err
// 	}

// 	return parseAIResponse(response, file.Path)
// }

// func buildPrompt(code string) string {
// 	prompt := fmt.Sprintf(`Analyze this code for security, performance, and maintainability issues.
	
// Code:
// %s

// Respond in JSON format:
// [{
// 	"line": <number>,
// 	"category": "security|performance|maintainability|error_handling",
// 	"description": "<concise issue description>",
// 	"severity": "high|medium|low",
// 	"suggestion": "<specific improvement suggestion>",
// 	"confidence": 0-1
// }]

// Rules:
// 1. Only report issues with confidence >= 0.7
// 2. Line numbers must be accurate
// 3. Suggest concrete fixes
// 4. Avoid trivial/style-only issues`, code)

// 	return prompt
// }

// func (g *GoogleAIClient) generateContent(ctx context.Context, prompt string) (string, error) {
// 	fmt.Println("Generating content with AI for prompt of length:", len(prompt))

// 	requestBody := map[string]interface{}{
// 		"contents": []map[string]interface{}{
// 			{
// 				"parts": []map[string]interface{}{
// 					{"text": prompt},
// 				},
// 			},
// 		},
// 		"generationConfig": map[string]interface{}{
// 			"temperature":     g.config.Temperature,
// 			"maxOutputTokens": g.config.MaxTokens,
// 		},
// 		"safetySettings": []map[string]interface{}{
// 			{
// 				"category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
// 				"threshold": "BLOCK_ONLY_HIGH",
// 			},
// 		},
// 	}

// 	jsonBody, _ := json.Marshal(requestBody)


// 	req, _ := http.NewRequestWithContext(ctx, "POST",
// 		fmt.Sprintf("%s?key=%s", googleAIEndpoint, g.apiKey),
// 		bytes.NewBuffer(jsonBody),
// 	)
// 	req.Header.Set("Content-Type", "application/json")

// 	resp, err := g.httpClient.Do(req)
// 	if err != nil {
// 		fmt.Println("API request failed:", err)
// 		return "", fmt.Errorf("API request failed: %w", err)
// 	}
// 	defer resp.Body.Close()

// 	if resp.StatusCode != http.StatusOK {
// 		body := readBody(resp)
// 		fmt.Println("API Error:", resp.StatusCode, body)
// 		return "", fmt.Errorf("API error %d: %s", resp.StatusCode, body)
// 	}

// 	var response struct {
// 		Candidates []struct {
// 			Content struct {
// 				Parts []struct {
// 					Text string `json:"text"`
// 				} `json:"parts"`
// 			} `json:"content"`
// 		} `json:"candidates"`
// 	}

// 	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
// 		fmt.Println("Failed to decode API response:", err)
// 		return "", fmt.Errorf("failed to decode response: %w", err)
// 	}

// 	if len(response.Candidates) == 0 {
// 		fmt.Println("No content in API response")
// 		return "", fmt.Errorf("no content in response")
// 	}

// 	return response.Candidates[0].Content.Parts[0].Text, nil
// }

// func parseAIResponse(response, filePath string) ([]*models.Issue, error) {


//     jsonStr := jsonRegex.FindString(response)
//     if jsonStr == "" {
//         fmt.Println("No JSON found in response")
//         return nil, fmt.Errorf("no JSON found in response")
//     }


//     if !strings.HasSuffix(strings.TrimSpace(jsonStr), "]") {
//         fmt.Println("Detected incomplete JSON, attempting to fix...")
//         jsonStr += "]" // Close the JSON array (basic fix)
//     }

//     var rawIssues []struct {
//         Line        int     `json:"line"`
//         Category    string  `json:"category"`
//         Description string  `json:"description"`
//         Severity    string  `json:"severity"`
//         Suggestion  string  `json:"suggestion"`
//         Confidence  float64 `json:"confidence"`
//     }

//     if err := json.Unmarshal([]byte(jsonStr), &rawIssues); err != nil {
//         fmt.Println("Invalid JSON format:", err)
//         return nil, fmt.Errorf("invalid JSON format: %w", err)
//     }

//     var issues []*models.Issue

//     for _, ri := range rawIssues {
//         if ri.Confidence < 0.7 {
//             continue
//         }

//         issues = append(issues, &models.Issue{
//             Path:        filePath,
//             Line:        ri.Line,
//             Title:       fmt.Sprintf("[%s] %s", strings.ToUpper(ri.Category), ri.Description),
//             Description: ri.Description,
//             Severity:    mapSeverity(ri.Severity),
//             Suggestion:  ri.Suggestion,
//             Source:      "AI Analysis",
//         })
//     }

//     return issues, nil
// }


// func mapSeverity(s string) models.Severity {
// 	switch strings.ToLower(s) {
// 	case "high":
// 		return models.SeverityError
// 	case "medium":
// 		return models.SeverityWarning
// 	default:
// 		return models.SeverityInfo
// 	}
// }

// func filterIssues(issues []*models.Issue, min models.Severity) []*models.Issue {
// 	var filtered []*models.Issue
// 	for _, issue := range issues {
// 		if issue.Severity >= min {
// 			filtered = append(filtered, issue)
// 		}
// 	}
// 	return filtered
// }

// func readBody(resp *http.Response) string {
// 	body, _ := io.ReadAll(resp.Body)
// 	return string(body)
// }
// ********************************************************************************************************************************

package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	// NOTE: You might need concurrency control if calling AnalyzeCode concurrently
	// "sync"
	// "github.com/keploy/keploy-review-agent/internal/analyzer/diff"
	"github.com/keploy/keploy-review-agent/pkg/models"
)

const (
	googleAIEndpoint      = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent" // Using 1.5 Flash - check if this is intended/available
	maxRetries            = 3
	baseDelay             = 1 * time.Second
	fixedContextLines     = 15  // Lines of context before/after hunk for fallback
	approxCharsPerToken   = 4   // Rough estimate for token calculation
	safetyTokenBuffer     = 500 // Buffer below model's max tokens (Gemini 1.5 Flash has a large context window, adjust buffer accordingly)
	// Default Max Input: Gemini 1.5 Flash often has 1M token context window, but API might impose lower practical limits.
	// Let's set a reasonably large default, but rely on the MaxOutputTokens calculation primarily.
	// This might need tuning based on API behavior and cost. Assume a large default for now.
	defaultMaxInputTokens = 20000
)

// Regex to find JSON array in LLM response
var jsonRegex = regexp.MustCompile(`(?s)\x60\x60\x60json\s*(\[[\s\S]*?\])\s*\x60\x60\x60|(\[[\s\S]*?{.*?}[\s\S]*?\])`)

// Regexes for Diff Parsing
var diffFileRegex = regexp.MustCompile(`^\+\+\+ b/(.*)`)
var diffHunkRegex = regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@`) // Captures new start line and new line count

// Regexes for Go Function Finding (Simplified - use go/parser for robustness)
// Allows for receiver name with pointer or non-pointer
var goFuncStartRegex = regexp.MustCompile(`^func(?:\s+\(\s*\*?\s*[a-zA-Z_]\w*\s+\*?\s*[a-zA-Z_][\w\.]*\s*\))?\s+([a-zA-Z_]\w*)\s*\(`)

type GoogleAIClient struct {
	apiKey         string
	httpClient     *http.Client
	config         *AIConfig
	maxInputTokens int // Calculated max input tokens allowed
}

type AIConfig struct {
	// MaxTokens renamed to MaxOutputTokens for clarity, as it refers to the *response* size limit.
	MaxOutputTokens int
	Temperature     float64
	MinSeverity     models.Severity
	// Consider adding ModelName string if you want to make the model configurable
	ModelName       string
}

// --- Data Structures for Diff Processing ---

type ChangedFile struct {
	Path       string
	Hunks      []*DiffHunk
	NewContent []string // Content of the new version of the file, split by lines
}

type DiffHunk struct {
	NewStartLine  int // Starting line number in the new file (1-based)
	NewLinesCount int // Number of lines in the hunk in the new file version
}

// Represents a piece of code sent to the LLM
type CodeSnippet struct {
	Content            string
	StartLineInFile    int // 1-based line number where this snippet starts in the original file
	OriginalHunkStart  int // Hunk start line *within this snippet* (1-based)
	OriginalHunkLength int // Hunk length *within this snippet*
}

func NewGoogleAIClient(apiKey string, cfg *AIConfig) *GoogleAIClient {
	// Use the configured model or default
	modelEndpoint := googleAIEndpoint // Default
	if cfg.ModelName != "" {
		// Construct endpoint based on name, assumes v1beta for now
		modelEndpoint = fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", cfg.ModelName)
	} else {
        log.Printf("No explicit model name configured, using default: %s", modelEndpoint)
    }

    // Estimate max input tokens based on common model limits (1M for 1.5 Flash is very large)
	// You might want a more conservative practical limit unless specifically needing huge context.
	// Using a large but not full 1M limit here.
    totalModelLimit := 1048576 // e.g., Gemini 1.5 Flash (1M tokens)
    
	// Set a practical upper bound for analysis to avoid excessive cost/latency, unless cfg indicates very large output needs.
    // Calculate based on MaxOutputTokens if it's set, otherwise use a reasonable default max input.
	maxInput := defaultMaxInputTokens
    if cfg.MaxOutputTokens > 0 {
	    calculatedMaxInput := totalModelLimit - cfg.MaxOutputTokens - safetyTokenBuffer
		if calculatedMaxInput > 0 {
			// Take the smaller of the calculated limit and our general default upper bound
            maxInput = min(defaultMaxInputTokens, calculatedMaxInput)
		} else {
			log.Printf("Warning: Calculated maxInputTokens (%d) based on MaxOutputTokens (%d) is too low. Using default: %d", calculatedMaxInput, cfg.MaxOutputTokens, defaultMaxInputTokens)
			maxInput = defaultMaxInputTokens
		}
    }


	log.Printf("Initializing GoogleAIClient. API Endpoint: %s, Max Output Tokens: %d, Calculated Max Input Tokens for API calls: %d", modelEndpoint, cfg.MaxOutputTokens, maxInput)

	return &GoogleAIClient{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 90 * time.Second, // Increased timeout for potentially larger analysis
		},
		config:         cfg,
		maxInputTokens: maxInput,
	}
}

// IMPORTANT: AnalyzeCode now requires the diff content and a map of file paths to their *new* content bytes.
// The caller (`o.runAnalyzer`) MUST be adapted to provide these instead of []*models.File.
//
// Parameters:
//   - ctx: Context for cancellation and deadlines.
//   - diffContent: The unified diff string output (e.g., from `git diff`).
//   - changedFilesContent: A map where keys are file paths (relative to repo root, matching diff output)
//     and values are the []byte content of the *new version* of those files.
func (g *GoogleAIClient) AnalyzeCode(ctx context.Context, diffContent string, changedFilesContent map[string][]byte) ([]*models.Issue, error) {
    log.Println("Starting AI code analysis based on diff...")

	if diffContent == "" {
		log.Println("AnalyzeCode called with empty diff content. Skipping AI analysis.")
		return []*models.Issue{}, nil // No diff, no issues
	}
	if len(changedFilesContent) == 0 {
		log.Println("AnalyzeCode called with empty file contents map. Skipping AI analysis as context is missing.")
        // Or return an error if content is expected when diff is present?
		return []*models.Issue{}, nil 
	}


	changedFiles, err := parseDiff(diffContent, changedFilesContent)
	if err != nil {
		return nil, fmt.Errorf("failed to parse diff: %w", err)
	}

    if len(changedFiles) == 0 {
        log.Println("Parsed diff but found no changed files requiring analysis.")
        return []*models.Issue{}, nil
    }


	var allIssues []*models.Issue
    // Consider using a channel and goroutines for concurrent analysis if performance is critical
    // var issuesCh = make(chan []*models.Issue)
    // var wg sync.WaitGroup

	for _, file := range changedFiles {
        // Example placeholder for potential concurrency:
        // wg.Add(1)
        // go func(f *ChangedFile) {
        //     defer wg.Done()
            // ... analysis logic ...
            // issuesCh <- analysisResult
        // }(file)


		// --- Existing Sequential Logic ---
		if shouldSkipFile(file.Path) {
			log.Printf("Skipping file based on extension: %s", file.Path)
			continue
		}

		log.Printf("Analyzing changes in file: %s", file.Path)
		issues, err := g.analyzeFileChanges(ctx, file)
		if err != nil {
			// Log error but continue with other files
			log.Printf("AI analysis skipped or failed for changes in %s: %v", file.Path, err)
			continue // Continue to the next file
		}

        // Filter issues by severity *before* adding them to the main list
		filtered := filterIssues(issues, g.config.MinSeverity)
        log.Printf("Found %d issues (after filtering at severity >= %s) for %s", len(filtered), g.config.MinSeverity, file.Path)
		if len(filtered) > 0 {
			// Use a mutex here if using goroutines
			allIssues = append(allIssues, filtered...)
		}
        // --- End Sequential Logic ---
	}

    // Example placeholder for collecting concurrent results:
    // go func() {
    //     wg.Wait()
    //     close(issuesCh)
    // }()
    // for result := range issuesCh {
    //      if len(result) > 0 {
	//		    filtered := filterIssues(result, g.config.MinSeverity) // Apply filtering here too
    //          allIssues = append(allIssues, filtered...)
    //      }
    // }

    // Deduplicate issues across all files/hunks
    deduplicatedIssues := deduplicateIssues(allIssues)
	log.Printf("AnalyzeCode: Completed analysis. Found %d total issues after deduplication.", len(deduplicatedIssues))
	return deduplicatedIssues, nil
}


// Parses the unified diff format and fetches file content lines
func parseDiff(diffContent string, fileContents map[string][]byte) ([]*ChangedFile, error) {
	var changedFiles []*ChangedFile
	var currentFile *ChangedFile = nil

	scanner := bufio.NewScanner(strings.NewReader(diffContent))
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "+++ b/") { // Detect new file info
			filePath := strings.TrimPrefix(line, "+++ b/")
			contentBytes, ok := fileContents[filePath]
			if !ok {
				log.Printf("Warning: Content for changed file '%s' not provided in map, skipping analysis for this file.", filePath)
				currentFile = nil // Invalidate currentFile until next '+++' line
				continue
			}
            if len(contentBytes) == 0 {
                log.Printf("Warning: Content for changed file '%s' is empty, skipping analysis for this file.", filePath)
                currentFile = nil
                continue
            }

			// Normalize line endings and split
			contentString := strings.ReplaceAll(string(contentBytes), "\r\n", "\n")
			currentFile = &ChangedFile{
				Path:       filePath,
				Hunks:      []*DiffHunk{},
				NewContent: strings.Split(contentString, "\n"),
			}
			changedFiles = append(changedFiles, currentFile)
			log.Printf("Parsing diff: Found changes for file: %s", filePath)
		} else if currentFile != nil { // Only process hunks if we are 'inside' a valid file block
			if matches := diffHunkRegex.FindStringSubmatch(line); len(matches) > 1 {
				startLine, _ := strconv.Atoi(matches[1])
                lineCount := 1 // Default hunk line count is 1 if not specified
                if len(matches) > 2 && matches[2] != "" {
                    count, err := strconv.Atoi(matches[2])
                    if err == nil && count >= 0 { // Allow count=0 for purely contextual hunks? Usually diff implies change. Let's assume >= 1 line usually. If 0, means no lines added.
                        lineCount = count // This count refers to the number of lines in the *new* file the hunk applies to.
                    }
                }

				hunk := &DiffHunk{
					NewStartLine:  startLine, // 1-based
					NewLinesCount: lineCount, // Number of lines in the new file this hunk definition represents
				}
				currentFile.Hunks = append(currentFile.Hunks, hunk)
                log.Printf("  - Found hunk affecting new file starting L%d, length %d lines", startLine, lineCount)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning diff content: %w", err)
	}

	if len(changedFiles) == 0 {
		log.Println("No changed files detected requiring analysis based on diff headers (+++ b/).")
	}

	return changedFiles, nil
}

// Analyzes changes within a single file by processing its hunks
func (g *GoogleAIClient) analyzeFileChanges(ctx context.Context, file *ChangedFile) ([]*models.Issue, error) {
	var fileIssues []*models.Issue

	for i, hunk := range file.Hunks {
		log.Printf("Processing hunk %d/%d in %s (affects new file from L%d)", i+1, len(file.Hunks), file.Path, hunk.NewStartLine)

        if hunk.NewStartLine > len(file.NewContent) || hunk.NewStartLine <= 0{
            log.Printf("  - Warning: Hunk start line %d is outside the bounds of the file content (length %d). Skipping hunk.", hunk.NewStartLine, len(file.NewContent))
            continue
        }

        // Try Function Context First (specific to Go for now)
		snippet, err := g.extractFunctionContextGo(file.NewContent, hunk.NewStartLine)
        useFunctionContext := (err == nil)

		if err != nil {
			if !errors.Is(err, errFunctionNotFound) { // Log specific finding errors, but not the simple "not found" case unless verbose
                 log.Printf("  - Detailed function context search error for hunk at L%d: %v", hunk.NewStartLine, err)
            }
			log.Printf("  - Could not find Go function context for hunk at L%d. Falling back to fixed-line context.", hunk.NewStartLine)
			// Fallback: Fixed Context
			snippet = g.extractFixedContext(file.NewContent, hunk.NewStartLine, hunk.NewLinesCount)
		} else {
			log.Printf("  - Extracted Go function context (lines %d-%d) for hunk at L%d", snippet.StartLineInFile, snippet.StartLineInFile+len(strings.Split(snippet.Content, "\n"))-1, hunk.NewStartLine)
		}

        // Estimate tokens for the chosen snippet + prompt overhead
		prompt := g.buildPromptForSnippet(snippet, file.Path, useFunctionContext)
		// Crude estimation - replace with tokenizer if accuracy is critical
		estimatedInputTokens := estimateTokens(snippet.Content) + estimateTokens(prompt) 
        
		log.Printf("  - Estimated input tokens for snippet + prompt: %d (Calculated Limit: %d)", estimatedInputTokens, g.maxInputTokens)

        // Check if combined tokens exceed limit
		if estimatedInputTokens >= g.maxInputTokens {
			// If function context was tried and failed due to size, fallback to fixed
			if useFunctionContext {
				log.Printf("  - Function context is too large (%d tokens >= %d limit). Falling back to fixed context.", estimatedInputTokens, g.maxInputTokens)
                useFunctionContext = false // Mark as using fixed context now
				snippet = g.extractFixedContext(file.NewContent, hunk.NewStartLine, hunk.NewLinesCount)
                prompt = g.buildPromptForSnippet(snippet, file.Path, useFunctionContext)
				estimatedInputTokens = estimateTokens(snippet.Content) + estimateTokens(prompt) // Recalculate
                
				log.Printf("  - Retrying with fixed context. Estimated tokens: %d", estimatedInputTokens)
				// Check fixed context size again
				if estimatedInputTokens >= g.maxInputTokens {
                    log.Printf("  - Fixed context is ALSO too large (%d tokens >= %d limit). Skipping this hunk.", estimatedInputTokens, g.maxInputTokens)
					continue // Skip this hunk entirely
                }
			} else {
                 // Already using fixed context and it's too large
                 log.Printf("  - Fixed context is too large (%d tokens >= %d limit). Skipping this hunk.", estimatedInputTokens, g.maxInputTokens)
                 continue // Skip this hunk entirely
            }
		}


		// Call the LLM with retry logic
        log.Printf("  - Calling LLM for hunk at L%d (using %s context)...", hunk.NewStartLine, map[bool]string{true:"function", false:"fixed-line"}[useFunctionContext])
		response, err := g.generateContentWithRetry(ctx, prompt)
		if err != nil {
			log.Printf("  - LLM call failed for hunk at L%d after retries: %v", hunk.NewStartLine, err)
			// Don't return error for the whole file, just skip this hunk's analysis
			continue
		}

		// Parse the response - Pass snippet start line for mapping relative lines back to file lines
		hunkIssues, err := parseAIResponse(response, file.Path, snippet.StartLineInFile)
		if err != nil {
			log.Printf("  - Failed to parse LLM response for hunk at L%d: %v", hunk.NewStartLine, err)
			continue // Skip issues from this hunk
		}

        log.Printf("  - Parsed %d valid issues from LLM for hunk at L%d", len(hunkIssues), hunk.NewStartLine)
		if len(hunkIssues) > 0 {
            fileIssues = append(fileIssues, hunkIssues...)
        }
	}

	return fileIssues, nil // Return successfully processed issues for the file
}

var errFunctionNotFound = errors.New("enclosing function definition not found")


// Tries to find the enclosing Go function for a change starting near targetLine (1-based).
// Returns the function snippet, its start line in the file (1-based), or error.
func (g *GoogleAIClient) extractFunctionContextGo(lines []string, targetLine int) (*CodeSnippet, error) {
	if targetLine <= 0 || targetLine > len(lines) {
		return nil, fmt.Errorf("targetLine %d out of bounds (1-%d)", targetLine, len(lines))
	}
    targetLine0 := targetLine - 1 // Use 0-based index internally

	funcDefLine := -1 // Line index containing "func [...] ("
    searchStartLine := targetLine0 // Start searching upwards from the target line

    // Search upwards for the 'func' keyword line
	for i := searchStartLine; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if goFuncStartRegex.MatchString(line) {
			// Basic check: Ensure it's not commented out
            if !strings.HasPrefix(line, "//") && !strings.HasPrefix(line, "/*") {
			    funcDefLine = i // Found a potential start
                break
            }
		}
        // Optimization: stop if we hit another potential func or package/import boundary? Maybe too complex for regex.
	}

	if funcDefLine == -1 {
		return nil, errFunctionNotFound // Use specific error
	}

    log.Printf("  - Potential Go function definition found at line %d (0-based) for target line %d", funcDefLine, targetLine0)

	// Find the matching closing brace '}' for the function block
	funcStartBodyLine := -1 // Line index of the opening brace '{'
	funcEndBodyLine := -1   // Line index of the matching closing brace '}'
	braceLevel := 0
	inBlockComment := false

	for i := funcDefLine; i < len(lines); i++ {
		line := lines[i]
        
        // Super basic block comment handling (only /* */ on single/multiple lines)
        // This is NOT robust against comments within code or nested block comments.
        if strings.Contains(line, "/*") { inBlockComment = true }
        if strings.Contains(line, "*/") { inBlockComment = false; continue /* Don't process rest of line with closing comment */}
        if inBlockComment { continue } // Skip lines entirely within block comment

		for j, char := range line {
             // Basic check for line comments
             if j > 0 && line[j-1] == '/' && char == '/' {
                  break // Stop processing rest of line
             }
            
			if char == '{' {
				if funcStartBodyLine == -1 { // Find the first opening brace on or after func def line
                    // More robustly, should find the first brace *after* the signature ()
                     funcStartBodyLine = i
				}
				braceLevel++
			} else if char == '}' {
				braceLevel--
			}
		}
		
        // Check if brace level returned to 0 *after* finding the opening brace
		if funcStartBodyLine != -1 && braceLevel == 0 {
            // Ensure the target line is actually within this identified function block
            if targetLine0 >= funcDefLine && targetLine0 <= i {
			    funcEndBodyLine = i // Found the matching brace for the func block containing the target
			    break
            } else {
                // We found a function block, but the target line is outside it.
                // Reset and continue searching in case of nested functions or functions defined before the target one.
                log.Printf("  - Found function block (%d-%d), but target line %d is outside. Continuing search.", funcDefLine, i, targetLine0)
                 funcStartBodyLine = -1 // Reset state
                 braceLevel = 0         // Reset brace level? Risky if parsing gets confused. Better to return not found here.
                return nil, fmt.Errorf("found function block L%d-L%d that does not contain target line %d", funcDefLine+1, i+1, targetLine)
                // Resetting might be complex. Let's return an error indicating mismatch.
                
            }
		}
	}

	if funcEndBodyLine == -1 {
		// Could not find matching brace OR target wasn't inside a completed block
		return nil, errors.New("could not find function block's matching closing '}' or target line was outside block")
	}
     
    // Safety check: Ensure start isn't after end (shouldn't happen here)
    if funcDefLine > funcEndBodyLine {
         return nil, fmt.Errorf("internal error: function start line %d > end line %d", funcDefLine+1, funcEndBodyLine+1)
    }

	// Extract snippet
	contentLines := lines[funcDefLine : funcEndBodyLine+1]

	// Calculate hunk position *within* the snippet
	// Hunk start line relative to the function definition line (1-based)
	hunkStartInSnippet := targetLine - funcDefLine

	snippet := &CodeSnippet{
		Content:            strings.Join(contentLines, "\n"),
		StartLineInFile:    funcDefLine + 1, // 1-based file line number
		OriginalHunkStart:  hunkStartInSnippet,
		// OriginalHunkLength needs to be passed down if needed for the prompt
		OriginalHunkLength: -1, // Placeholder
	}

	return snippet, nil
}

// Extracts a fixed number of lines around the changed hunk area.
func (g *GoogleAIClient) extractFixedContext(lines []string, hunkStartLine, hunkLinesCount int) *CodeSnippet {
    // Use 0-based indexing for slicing
    hunkStart0 := hunkStartLine - 1
    
    // Determine the actual end line affected by the hunk in 0-based index
    // Diff hunk length can be 0 if only lines were deleted. We still need context.
    // If count is 0, the "affected area" is just before the start line conceptually.
    // If count > 0, the last line affected is start + count - 1.
    hunkEnd0 := hunkStart0
    if hunkLinesCount > 0 {
        hunkEnd0 = hunkStart0 + hunkLinesCount -1
    }

    // Calculate context window start/end (0-based)
	start0 := max(0, hunkStart0-fixedContextLines)
	end0 := min(len(lines)-1, hunkEnd0+fixedContextLines) 

    // Ensure start isn't after end, adjust if necessary
    if start0 > end0 {
       log.Printf("Warning: Fixed context calculation resulted in start (%d) > end (%d). Adjusting.", start0, end0)
       if start0 < len(lines) { // If start is valid, set end=start
           end0 = start0
       } else if len(lines) > 0 { // If start was out of bounds, pull both back to last valid index
            start0 = len(lines)-1
            end0 = start0
       } else { // Empty file?
            start0 = 0
            end0 = -1 // Represents an empty slice
       }
    }
	
    // Extract content lines, handle empty slice case
    var contentLines []string
    if end0 >= start0 {
        contentLines = lines[start0 : end0+1]
    } else {
         contentLines = []string{} // Empty content
    }

    // Hunk start line relative to the start of the *snippet* (1-based)
    // If snippet starts at line 5 (file), and hunk starts at line 10 (file)
    // hunkStartInSnippet = 10 - 5 = 5. (Hunk starts on line 5 of the snippet)
	hunkStartInSnippet := hunkStartLine - (start0 +1) + 1 // +1 to convert back to 1-based relative line

	snippet := &CodeSnippet{
		Content:            strings.Join(contentLines, "\n"),
		StartLineInFile:    start0 + 1, // 1-based start line in the original file
		OriginalHunkStart:  hunkStartInSnippet,
		OriginalHunkLength: hunkLinesCount, // Keep track of the original hunk size
	}
	return snippet
}

// Builds the prompt for the LLM, incorporating the code snippet and context info
func (g *GoogleAIClient) buildPromptForSnippet(snippet *CodeSnippet, filePath string, isFunctionContext bool) string {
	contextType := "code snippet"
	focusInstruction := fmt.Sprintf("The changes occurred around line %d within this provided %s.", snippet.OriginalHunkStart, contextType)
	if isFunctionContext {
		contextType = "Go function" // Customize if supporting other languages
		focusInstruction = fmt.Sprintf("Pay special attention to the logic around line %d within this %s, which corresponds to the recent changes.", snippet.OriginalHunkStart, contextType)
	}
    if snippet.OriginalHunkStart <=0 { // Add robustness if calculation failed
        focusInstruction = "Changes are contained within this snippet."
    }

    // Choose model based on config if available
    modelName := "Gemini" // Default
	if g.config.ModelName != "" {
		modelName = g.config.ModelName // e.g., "Gemini 1.5 Flash"
	}


	// Using Markdown code blocks for JSON output as recommended by Google for Gemini 1.5
	prompt := fmt.Sprintf(`You are a code review assistant based on %s.
Analyze the following %s from the file '%s'.
%s

Code %s (Lines %d-%d of %s):
--- START CODE ---
%s
--- END CODE ---

Your Task:
Identify potential issues related to security, performance, maintainability, reliability, and error handling in the code provided above.

Response Format Rules:
1. Respond ONLY with a valid JSON array enclosed in Markdown code fences (\x60\x60\x60json ... \x60\x60\x60).
2. The JSON array should contain objects, each representing a single issue found.
3. Each issue object MUST have the following keys:
   - "line": <integer, the line number where the issue occurs, relative to the START of the Code Snippet provided above (snippet starts at line 1)>,
   - "category": <string, one of: "security", "performance", "maintainability", "reliability", "error_handling">,
   - "description": "<string, concise explanation of the issue and its potential impact>",
   - "severity": "<string, one of: "high", "medium", "low">,
   - "suggestion": "<string, actionable recommendation or code example for fixing the issue>",
   - "confidence": <float, your confidence level (0.0 to 1.0) that this is a genuine issue>

Analysis Guidelines:
- Focus on substantive programming issues. Avoid purely stylistic comments (like code formatting).
- Line numbers MUST be accurate relative to the start of the provided Code Snippet.
- Ensure 'description' clearly states the problem and 'suggestion' provides a concrete fix.
- Severity: 'high' for critical risks/bugs, 'medium' for significant issues, 'low' for best-practice violations or minor improvements.
- Only include issues with confidence >= 0.7.
- If no issues with confidence >= 0.7 are found, respond with an empty JSON array: \x60\x60\x60json[]\x60\x60\x60.
- Do not include any text outside the \x60\x60\x60json ... \x60\x60\x60 block.`,
		modelName,
		contextType,
		filepath.Base(filePath), // Use Base path for brevity
		focusInstruction,
		contextType,
        snippet.StartLineInFile, // Indicate original line numbers for context
        snippet.StartLineInFile + max(0, len(strings.Split(snippet.Content, "\n"))-1), // Calculate end line
		filepath.Base(filePath),
		snippet.Content,
	)

	return prompt
}


// generateContentWithRetry handles the API call and retry logic
func (g *GoogleAIClient) generateContentWithRetry(ctx context.Context, prompt string) (string, error) {
	var response string
	var lastErr error
	for i := 0; i < maxRetries; i++ {
        attemptCtx := ctx // Inherit timeout from caller for the whole retry sequence? Or use per-attempt below?
        // Per-attempt timeout might be better:
        // attemptCtx, cancel := context.WithTimeout(ctx, g.httpClient.Timeout) 

        var attemptErr error
        response, attemptErr = g.generateContent(attemptCtx, prompt)
        // cancel() // Use if using per-attempt timeout

		if attemptErr == nil {
			return response, nil // Success
		}
        lastErr = attemptErr // Store the last error encountered

		// Check for non-retryable conditions
		if errors.Is(attemptErr, context.DeadlineExceeded) || errors.Is(attemptErr, context.Canceled) {
			log.Printf("Context cancelled or deadline exceeded during attempt %d: %v. Not retrying.", i+1, attemptErr)
			return "", attemptErr // Don't retry on context errors
		}
		var sce *statusCodeError
		if errors.As(attemptErr, &sce) {
			if sce.StatusCode >= 400 && sce.StatusCode < 500 && sce.StatusCode != 429 { // Don't retry client errors except rate limiting (429)
				log.Printf("Client error %d during attempt %d. Not retrying: %v", sce.StatusCode, i+1, attemptErr)
				return "", attemptErr
			}
             // Consider specific handling for 429 Too Many Requests (maybe longer, specific delay)
             if sce.StatusCode == 429 {
                 log.Printf("Rate limit hit (429) during attempt %d. Applying retry delay.", i+1)
             }
		}
        // Check for specific Google AI errors (e.g., prompt blocked) which might not be retryable
         // Example: Assuming blocked prompts return a specific error message/code in statusCodeError body
        if sce != nil && (strings.Contains(sce.Body, "prompt blocked") || strings.Contains(sce.Body,"SAFETY")) {
            log.Printf("Prompt or response blocked due to safety settings during attempt %d. Not retrying: %v", i+1, attemptErr)
             return "", attemptErr // Don't retry safety blocks
        }

		// Don't sleep on the last attempt
		if i == maxRetries-1 {
			break
		}

		// Exponential backoff: 1s, 2s, 4s... + some jitter?
		delay := baseDelay * time.Duration(1<<i)
		log.Printf("Attempt %d/%d failed for LLM call. Retrying in %v due to error: %v", i+1, maxRetries, delay, attemptErr)
		select {
		case <-time.After(delay):
			// Continue to next iteration
		case <-ctx.Done():
			log.Printf("Overall context cancelled while waiting to retry: %v", ctx.Err())
			return "", fmt.Errorf("retry interrupted by context cancellation: %w (last error: %v)", ctx.Err(), lastErr)
		}
	}
	return "", fmt.Errorf("failed after %d retries: %w", maxRetries, lastErr) // Return the last error encountered
}


// generateContent makes the actual HTTP call to the Google AI API
func (g *GoogleAIClient) generateContent(ctx context.Context, prompt string) (string, error) {
	apiURL := g.getAPIEndpoint() // Use helper to get correct endpoint
	log.Printf("Calling Google AI API: %s (Prompt length: %d chars)", apiURL, len(prompt))


    generationConfig := map[string]interface{}{
		"temperature": g.config.Temperature,
        // MaxOutputTokens should be set if configured, otherwise rely on model defaults perhaps?
		// Setting it explicitly based on config is usually better.
	}
    if g.config.MaxOutputTokens > 0 {
        generationConfig["maxOutputTokens"] = g.config.MaxOutputTokens
		// For JSON output, adding response_mime_type can be helpful if supported by the model/API version
        // generationConfig["responseMimeType"] = "application/json" // Requires testing if model supports it well
    }

	requestBodyMap := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"parts": []map[string]interface{}{
					{"text": prompt},
				},
			},
		},
		"generationConfig": generationConfig,
		// Updated Safety Settings - BLOCK_MEDIUM_AND_ABOVE might be too strict for code analysis? Consider BLOCK_ONLY_HIGH or NONE. Test this carefully.
		// Let's try BLOCK_MEDIUM_AND_ABOVE as a safe default.
		"safetySettings": []map[string]interface{}{ 
			{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
			{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
			{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
			{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
		},
	}

	jsonBody, err := json.Marshal(requestBodyMap)
	if err != nil {
		// This should not happen with a map[string]interface{}
		log.Printf("CRITICAL: Failed to marshal request body: %v", err)
		return "", fmt.Errorf("internal error: failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", fmt.Sprintf("%s?key=%s", apiURL, g.apiKey), bytes.NewBuffer(jsonBody))
	if err != nil {
		log.Printf("CRITICAL: Failed to create HTTP request: %v", err)
		return "", fmt.Errorf("internal error: failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json") // Explicitly accept JSON

	resp, err := g.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("API request failed: %w", err) // Network error, timeout, etc.
	}
	defer resp.Body.Close()

	bodyBytes, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		log.Printf("Failed to read response body (Status %d): %v", resp.StatusCode, readErr)
		// Try to return status code error even if body reading fails
		if resp.StatusCode != http.StatusOK {
			return "", &statusCodeError{StatusCode: resp.StatusCode, Body: fmt.Sprintf("Status %d (failed to read body: %v)", resp.StatusCode, readErr)}
		}
		return "", fmt.Errorf("failed to read response body: %w", readErr)
	}
	bodyString := string(bodyBytes) // Keep for logging on error


	if resp.StatusCode != http.StatusOK {
        log.Printf("API request failed: Status %d, Body: %s", resp.StatusCode, bodyString)
		// Attempt to parse standard Google AI error structure
		var googleErrResp struct {
			Error struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
				Status  string `json:"status"`
                Details []interface{} `json:"details"` // Capture details if present
			} `json:"error"`
		}
		if json.Unmarshal(bodyBytes, &googleErrResp) == nil && googleErrResp.Error.Message != "" {
			log.Printf("Google AI API Error Details: Code=%d, Status=%s, Message=%s", googleErrResp.Error.Code, googleErrResp.Error.Status, googleErrResp.Error.Message)
			return "", &statusCodeError{StatusCode: resp.StatusCode, Body: fmt.Sprintf("Code=%d, Status=%s: %s", googleErrResp.Error.Code, googleErrResp.Error.Status, googleErrResp.Error.Message)}
		}
		// Fallback to generic status code error if parsing specific structure fails
		return "", &statusCodeError{StatusCode: resp.StatusCode, Body: bodyString}
	}

	// Decode the expected successful response structure
	var responsePayload struct {
		Candidates []struct {
			Content *struct { // Pointer to handle null content?
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
				Role string `json:"role"`
			} `json:"content"`
			FinishReason  string `json:"finishReason"`
			SafetyRatings []struct {
				Category    string `json:"category"`
				Probability string `json:"probability"`
				Blocked     *bool  `json:"blocked,omitempty"` // Check if explicit block indicated
			} `json:"safetyRatings"`
			CitationMetadata *struct{
                 CitationSources []interface{} `json:"citationSources"`
            } `json:"citationMetadata,omitempty"`
            TokenCount int `json:"tokenCount"` // Output token count
		} `json:"candidates"`
		UsageMetadata *struct { // Overall token counts
			PromptTokenCount int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
        PromptFeedback *struct { // Check for prompt blocks
			BlockReason   string `json:"blockReason"`
			SafetyRatings []struct {
				Category    string `json:"category"`
				Probability string `json:"probability"`
			} `json:"safetyRatings"`
		} `json:"promptFeedback,omitempty"`
	}


	if err := json.Unmarshal(bodyBytes, &responsePayload); err != nil {
		log.Printf("Failed to decode successful API response JSON (Status %d): %v\nRaw Body:\n%s", resp.StatusCode, err, bodyString)
		// Sometimes Gemini might return just text even on 200 OK if something went wrong. Check for that.
        // If body doesn't look like JSON, maybe return it as an error message?
		if !strings.HasPrefix(strings.TrimSpace(bodyString), "{") {
            return "", fmt.Errorf("API returned non-JSON success response (Status %d): %s", resp.StatusCode, bodyString)
        }
		return "", fmt.Errorf("failed to decode valid JSON from API response: %w", err)
	}

	// Log Token Usage
	if responsePayload.UsageMetadata != nil {
        log.Printf("Token Usage: Prompt=%d, Candidates=%d, Total=%d", responsePayload.UsageMetadata.PromptTokenCount, responsePayload.UsageMetadata.CandidatesTokenCount, responsePayload.UsageMetadata.TotalTokenCount)
    } else if len(responsePayload.Candidates) > 0 && responsePayload.Candidates[0].TokenCount > 0 {
        // Fallback if only candidate token count is available
        log.Printf("Token Usage: Candidates=%d", responsePayload.Candidates[0].TokenCount)
    }

	// Check for Prompt Feedback/Blocks
	if responsePayload.PromptFeedback != nil && responsePayload.PromptFeedback.BlockReason != "" {
		log.Printf("Error: Prompt was blocked by Google AI. Reason: %s", responsePayload.PromptFeedback.BlockReason)
		// Provide details from safety ratings if available
        details := ""
         for _, rating := range responsePayload.PromptFeedback.SafetyRatings {
             details += fmt.Sprintf(" [%s: %s]", rating.Category, rating.Probability)
         }
		return "", fmt.Errorf("prompt blocked by safety settings (Reason: %s)%s", responsePayload.PromptFeedback.BlockReason, details)
	}

	// Check for empty or blocked candidates
	if len(responsePayload.Candidates) == 0 {
        log.Printf("Warning: No candidates returned in API response. Finish Reason: N/A. Prompt Feedback: %+v", responsePayload.PromptFeedback)
		return "", errors.New("no candidates found in API response")
	}

    candidate := responsePayload.Candidates[0]
    finishReason := candidate.FinishReason
    log.Printf("Candidate Finish Reason: %s", finishReason)

    // Check finish reason for non-OK completion
	if finishReason == "STOP" {
        // STOP is usually the normal completion
	} else if finishReason == "MAX_TOKENS" {
		log.Printf("Warning: Response possibly truncated by maxOutputTokens limit.")
        // Proceed, but be aware output might be incomplete
	} else if finishReason == "SAFETY" {
        details := ""
        for _, rating := range candidate.SafetyRatings {
             isBlocked := false
             if rating.Blocked != nil { isBlocked = *rating.Blocked }
            details += fmt.Sprintf(" [%s: %s, Blocked: %t]", rating.Category, rating.Probability, isBlocked)
        }
		log.Printf("Error: Response blocked due to safety settings. Finish Reason: %s. Details:%s", finishReason, details)
		return "", fmt.Errorf("response blocked by safety settings (Reason: %s)%s", finishReason, details)
	} else if finishReason == "RECITATION" {
         log.Printf("Warning: Response likely blocked or modified due to potential recitation. Finish Reason: %s", finishReason)
        // Consider if this should be an error or allow partial response
         return "", fmt.Errorf("response stopped due to potential recitation (Reason: %s)", finishReason)
    } else if finishReason != "" && finishReason != "FINISH_REASON_UNSPECIFIED" && finishReason != "STOP" {
         // Catch other unexpected finish reasons
        log.Printf("Warning: Unexpected finish reason '%s'.", finishReason)
         // Proceed but log it
    }


	// Check actual content presence
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 || candidate.Content.Parts[0].Text == "" {
		log.Printf("Warning: Candidate has no usable text content returned. Finish Reason: %s. Safety Ratings: %+v", finishReason, candidate.SafetyRatings)
		// Check if safety blocked despite finishReason != "SAFETY"
		for _, rating := range candidate.SafetyRatings {
			if rating.Blocked != nil && *rating.Blocked {
				return "", fmt.Errorf("response content missing, likely due to safety block (Category: %s, Probability: %s)", rating.Category, rating.Probability)
			}
		}
		return "", fmt.Errorf("no text content found in the first candidate part (Finish Reason: %s)", finishReason)
	}

	// Return the text from the first part
	llmTextOutput := candidate.Content.Parts[0].Text
	log.Printf("LLM Response: %s\n", llmTextOutput)
	log.Printf("LLM Raw Output Length: %d chars", len(llmTextOutput))
	return llmTextOutput, nil
}


// Parses the AI's JSON response, maps line numbers, and filters by confidence.
func parseAIResponse(response string, filePath string, snippetStartLine int) ([]*models.Issue, error) {
	// Extract the JSON part, potentially enclosed in Markdown fences
	var jsonStr string
    matches := jsonRegex.FindStringSubmatch(response)

    if len(matches) > 2 && matches[1] != "" { // Check group 1 first (```json [...] ```)
        jsonStr = matches[1]
        log.Printf("Extracted JSON using Markdown fences.")
	} else if len(matches) > 2 && matches[2] != "" { // Check group 2 ([...])
		 jsonStr = matches[2]
        log.Printf("Extracted JSON using simple array match.")
    } else {
		log.Printf("No valid JSON array found in LLM response for %s (snippet starting L%d).\nRaw response was:\n---\n%s\n---", filePath, snippetStartLine, response)
		return nil, errors.New("no JSON array found in LLM response")
	}

	// Attempt to unmarshal the extracted JSON string
	var rawIssues []struct {
		Line        interface{} `json:"line"` // Accept flexible type for line number initially
		Category    string      `json:"category"`
		Description string      `json:"description"`
		Severity    string      `json:"severity"`
		Suggestion  string      `json:"suggestion"`
		Confidence  float64     `json:"confidence"`
	}

	decoder := json.NewDecoder(strings.NewReader(jsonStr))
    // decoder.DisallowUnknownFields() // Optional: Make parsing stricter

	if err := decoder.Decode(&rawIssues); err != nil {
		log.Printf("Invalid JSON format received from LLM for %s (snippet starting L%d): %v\nProblematic JSON string extracted:\n---\n%s\n---", filePath, snippetStartLine, err, jsonStr)
		
		// Try simple cleaning heuristics (example: remove trailing comma before ']') - Often risky
        // cleanedJson := strings.TrimSpace(jsonStr)
        // if strings.HasSuffix(cleanedJson, ",]") {
        //     cleanedJson = cleanedJson[:len(cleanedJson)-2] + "]"
        //      log.Println("Attempting cleanup: Removed trailing comma in JSON array.")
        //      if errRetry := json.Unmarshal([]byte(cleanedJson), &rawIssues); errRetry == nil {
        //          log.Println("JSON parsing succeeded after cleanup.")
        //          err = nil // Clear the error
        //      } else {
        //           log.Printf("Cleanup failed: %v", errRetry)
        //      }
        // }
         // If still error after potential cleanup
         if err != nil {
		    return nil, fmt.Errorf("invalid JSON format from LLM: %w", err)
        }
	}

	var issues []*models.Issue
	validIssuesCount := 0
	skippedConfidence := 0
    skippedInvalid := 0
	for _, ri := range rawIssues {
		// Validate and Convert Line Number
		var lineInt int
		switch v := ri.Line.(type) {
		case float64:
            lineInt = int(v) // Convert float (common from JSON numbers)
            if float64(lineInt) != v {
                 log.Printf("Warning: LLM returned non-integer line number (%v) for issue: '%s'. Skipping.", ri.Line, ri.Description)
                 skippedInvalid++
                 continue
            }
		case int:
			lineInt = v
        case string: // Allow string if it parses to int
             var err error
             lineInt, err = strconv.Atoi(v)
             if err != nil {
                log.Printf("Warning: LLM returned non-numeric string line number ('%s') for issue: '%s'. Skipping.", ri.Line, ri.Description)
                skippedInvalid++
                 continue
            }
		default:
            log.Printf("Warning: LLM returned unexpected type for line number (%T: %v) for issue: '%s'. Skipping.", ri.Line, ri.Line, ri.Description)
             skippedInvalid++
			continue
		}


        if lineInt <= 0 {
             log.Printf("Warning: Skipping issue with invalid relative line number <= 0 (line=%d): %s", lineInt, ri.Description)
             skippedInvalid++
             continue
         }
         
        // Apply confidence filter *after* validating line number
		if ri.Confidence < 0.7 {
            skippedConfidence++
			continue
		}


		// Map relative line number to absolute file line number
        // snippetStartLine is 1-based, ri.Line is 1-based relative to snippet
		absoluteLine := lineInt + snippetStartLine - 1

        // Basic validation of other fields
		category := strings.ToLower(strings.TrimSpace(ri.Category))
        validCategories := map[string]bool{"security": true, "performance": true, "maintainability": true, "reliability": true, "error_handling": true}
        if !validCategories[category] {
             log.Printf("Warning: Skipping issue with invalid category '%s'. Defaulting to 'general'. Desc: %s", ri.Category, ri.Description)
            // category = "general" // Or skip? Skipping seems safer if format is wrong. Let's skip.
            skippedInvalid++
            continue
        }

		severity := mapSeverity(ri.Severity) // Handles mapping and defaulting
        description := strings.TrimSpace(ri.Description)
        suggestion := strings.TrimSpace(ri.Suggestion)

        if description == "" || suggestion == "" {
             log.Printf("Warning: Skipping issue with missing description or suggestion (Line %d): Cat: %s, Desc: '%s', Suggest: '%s'", absoluteLine, category, description, suggestion)
             skippedInvalid++
            continue
        }

		issues = append(issues, &models.Issue{
			Path:        filePath,
			Line:        absoluteLine,
			// Use more structured Title/Desc? Maybe just Category and Description raw.
			Title:       fmt.Sprintf("[%s] %s", strings.ToUpper(category[:1])+category[1:], description), // Capitalize category
			Description: description,
			Severity:    severity,
			Suggestion:  suggestion,
			Source:      "AI Analysis",
			// Confidence: ri.Confidence, // Add confidence to Issue model if desired
		})
		validIssuesCount++
	}

    log.Printf("Parsed LLM Response for %s (snippet L%d): Total Raw Issues: %d, Valid Issues Found: %d, Skipped (Low Confidence): %d, Skipped (Invalid Format): %d",
		filePath, snippetStartLine, len(rawIssues), validIssuesCount, skippedConfidence, skippedInvalid)

    // If JSON parsed but resulted in zero valid issues (e.g., all filtered out)
    if len(rawIssues) > 0 && validIssuesCount == 0 {
         log.Printf("  - Note: JSON from LLM was parsed, but yielded zero issues after validation/filtering.")
    }

	return issues, nil
}


// --- Helper Functions ---

func (g *GoogleAIClient) getAPIEndpoint() string {
     if g.config.ModelName != "" {
		return fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent", g.config.ModelName)
	}
	return googleAIEndpoint // Default
}

// Simple token estimation - Replace with a library specific to the model (e.g., tiktoken for OpenAI) if accuracy matters.
// Gemini tokenization might differ. This is a *very* rough heuristic.
func estimateTokens(text string) int {
    // Count characters roughly, divide by average chars per token.
	// Also add a small overhead per text segment for model processing.
	overhead := 10
	return (len(text) / approxCharsPerToken) + overhead
}


func shouldSkipFile(path string) bool {
	ext := filepath.Ext(path)
	// Focused list of extensions - make this configurable?
	supported := map[string]bool{
		".go": true,
		".py": true,
		".js": true,
		".ts": true,
        // Add other languages as needed
	}
	_, isSupported := supported[ext]

	// Add checks for common non-code files or paths to skip
	if strings.Contains(path, "vendor/") || // Go vendor dir
       strings.Contains(path, "node_modules/") || // Node modules
       strings.HasSuffix(path, "_test.go") || // Go test files (optional - might want to analyze?)
       strings.Contains(path, "/test/") || // General test dirs
       strings.Contains(path, "/tests/") || // General test dirs
       strings.HasSuffix(path, ".min.js") || // Minified JS
       strings.HasPrefix(path, "dist/") || // Common build output dirs
       strings.HasPrefix(path, "build/") ||
       filepath.Base(path) == "go.mod" || // Go module files
       filepath.Base(path) == "go.sum" ||
       filepath.Base(path) == "package.json" || // Package manager files
       filepath.Base(path) == "package-lock.json" ||
       filepath.Base(path) == "yarn.lock" ||
       filepath.Base(path) == "requirements.txt" {
		
		return true
	}


	if !isSupported {
        log.Printf("Debug: Skipping file '%s', unsupported extension '%s'", path, ext)
        return true
    }

	return false // Not skipped
}

func mapSeverity(s string) models.Severity {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "high", "critical", "error": // Allow synonyms
		return models.SeverityError
	case "medium", "moderate", "warning": // Allow synonyms
		return models.SeverityWarning
	case "low", "minor", "info", "information": // Allow synonyms
		return models.SeverityInfo
	default:
		log.Printf("Warning: Unrecognized severity '%s' received from LLM, defaulting to Info", s)
		return models.SeverityInfo
	}
}

func filterIssues(issues []*models.Issue, min models.Severity) []*models.Issue {
	if min == models.SeverityWarning{ // Allow filtering only if a valid minimum is set
		return issues
	}
	var filtered []*models.Issue
	for _, issue := range issues {
		if issue.Severity >= min {
			filtered = append(filtered, issue)
		}
	}
	return filtered
}

// Deduplicates issues based on File, Line, and Title.
func deduplicateIssues(issues []*models.Issue) []*models.Issue {
    if len(issues) <= 1 { return issues } // No need to deduplicate

	seen := make(map[string]bool)
	var deduped []*models.Issue
    duplicateCount := 0
	for _, issue := range issues {
		// Create a key. Using Title might be too broad if descriptions are slightly different for same line.
        // Consider using category + line + file path ? Or hash of description?
        // Let's stick to File:Line:Title for now.
		key := fmt.Sprintf("%s::%d::%s", issue.Path, issue.Line, issue.Title)

		if !seen[key] {
			seen[key] = true
			deduped = append(deduped, issue)
		} else {
             // Log duplicate only once perhaps?
             if duplicateCount < 10 { // Limit excessive logging
                 log.Printf("Deduplicating redundant issue: %s at %s:%d", issue.Title, issue.Path, issue.Line)
             } else if duplicateCount == 10 {
                 log.Println("Further duplicate issue logs suppressed.")
             }
             duplicateCount++
        }
	}
    if duplicateCount > 0 {
         log.Printf("Deduplicated %d redundant issues.", duplicateCount)
    }
	return deduped
}

// max helper
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min helper
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Custom error type to hold status code and body for API errors
type statusCodeError struct {
	StatusCode int
	Body       string
}

func (e *statusCodeError) Error() string {
    // Limit body length in error message for readability
    maxBodyLen := 200
    bodySnippet := e.Body
    if len(bodySnippet) > maxBodyLen {
        bodySnippet = bodySnippet[:maxBodyLen] + "..."
    }
	return fmt.Sprintf("API error status %d: %s", e.StatusCode, bodySnippet)
}
