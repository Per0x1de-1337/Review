package llm

import (
	"context"
	"fmt"
	"strings"

	"github.com/keploy/keploy-review-agent/internal/config"
	"github.com/keploy/keploy-review-agent/pkg/models"
)

// Engine implements the LLM-based code analysis
type Engine struct {
	cfg *config.Config
}
