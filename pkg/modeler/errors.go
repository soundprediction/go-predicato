package modeler

import "fmt"

// ModelerErrorHandling controls behavior when a custom GraphModeler returns an error.
type ModelerErrorHandling int

const (
	// FailOnError stops processing and returns the error immediately.
	// Use this when you want strict validation of custom modeler behavior.
	FailOnError ModelerErrorHandling = iota

	// FallbackOnError logs a warning and uses DefaultModeler for the failed step.
	// This is the default behavior, providing resilience while alerting to issues.
	FallbackOnError

	// SkipOnError logs a warning and skips the failed step entirely.
	// Use with caution - may result in incomplete graph modeling.
	SkipOnError
)

// String returns the string representation of the error handling mode.
func (m ModelerErrorHandling) String() string {
	switch m {
	case FailOnError:
		return "FailOnError"
	case FallbackOnError:
		return "FallbackOnError"
	case SkipOnError:
		return "SkipOnError"
	default:
		return fmt.Sprintf("ModelerErrorHandling(%d)", m)
	}
}

// ModelerError wraps an error with additional context about which modeler step failed.
type ModelerError struct {

	// Err is the underlying error
	Err error

	// Step is which modeler method failed (e.g., "ResolveEntities")
	Step string

	// Fallback indicates whether fallback to DefaultModeler was used
	Fallback bool

	// Skipped indicates whether the step was skipped
	Skipped bool
}

// Error implements the error interface.
func (e *ModelerError) Error() string {
	if e.Fallback {
		return fmt.Sprintf("modeler %s failed (using fallback): %v", e.Step, e.Err)
	}
	if e.Skipped {
		return fmt.Sprintf("modeler %s failed (skipped): %v", e.Step, e.Err)
	}
	return fmt.Sprintf("modeler %s failed: %v", e.Step, e.Err)
}

// Unwrap returns the underlying error.
func (e *ModelerError) Unwrap() error {
	return e.Err
}

// NewModelerError creates a new ModelerError.
func NewModelerError(step string, err error) *ModelerError {
	return &ModelerError{
		Step: step,
		Err:  err,
	}
}

// WithFallback marks this error as having used fallback.
func (e *ModelerError) WithFallback() *ModelerError {
	e.Fallback = true
	return e
}

// WithSkipped marks this error as having been skipped.
func (e *ModelerError) WithSkipped() *ModelerError {
	e.Skipped = true
	return e
}
