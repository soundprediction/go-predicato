package driver

import (
	"testing"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j/dbtype"
)

func TestTypeConversionError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		err      *TypeConversionError
		expected string
	}{
		{
			name: "with field",
			err: &TypeConversionError{
				Expected: "string",
				Actual:   "int64",
				Field:    "node_id",
			},
			expected: `type conversion error for field "node_id": expected string, got int64`,
		},
		{
			name: "without field",
			err: &TypeConversionError{
				Expected: "dbtype.Node",
				Actual:   "nil",
				Field:    "",
			},
			expected: "type conversion error: expected dbtype.Node, got nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := tt.err.Error(); got != tt.expected {
				t.Errorf("Error() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestAsString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  any
		want   string
		wantOK bool
	}{
		{"valid string", "hello", "hello", true},
		{"empty string", "", "", true},
		{"nil", nil, "", false},
		{"int", 42, "", false},
		{"float", 3.14, "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsString(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsString() ok = %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Errorf("AsString() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAsInt64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  any
		want   int64
		wantOK bool
	}{
		{"valid int64", int64(42), 42, true},
		{"zero", int64(0), 0, true},
		{"negative", int64(-100), -100, true},
		{"nil", nil, 0, false},
		{"int (wrong type)", 42, 0, false},
		{"string", "42", 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsInt64(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsInt64() ok = %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Errorf("AsInt64() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestAsFloat64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  any
		want   float64
		wantOK bool
	}{
		{"valid float64", float64(3.14), 3.14, true},
		{"zero", float64(0), 0, true},
		{"nil", nil, 0, false},
		{"int64 (wrong type)", int64(42), 0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsFloat64(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsFloat64() ok = %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Errorf("AsFloat64() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestAsBool(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  any
		want   bool
		wantOK bool
	}{
		{"true", true, true, true},
		{"false", false, false, true},
		{"nil", nil, false, false},
		{"string", "true", false, false},
		{"int", 1, false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsBool(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsBool() ok = %v, want %v", ok, tt.wantOK)
			}
			if got != tt.want {
				t.Errorf("AsBool() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAsDBNode(t *testing.T) {
	t.Parallel()

	validNode := dbtype.Node{
		ElementId: "123",
		Labels:    []string{"Entity"},
		Props:     map[string]any{"uuid": "test-uuid"},
	}

	tests := []struct {
		name   string
		input  any
		wantOK bool
	}{
		{"valid node", validNode, true},
		{"nil", nil, false},
		{"string", "not a node", false},
		{"map", map[string]any{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsDBNode(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsDBNode() ok = %v, want %v", ok, tt.wantOK)
			}
			if tt.wantOK && got.ElementId != validNode.ElementId {
				t.Errorf("AsDBNode() ElementId = %s, want %s", got.ElementId, validNode.ElementId)
			}
		})
	}
}

func TestAsDBRelationship(t *testing.T) {
	t.Parallel()

	validRel := dbtype.Relationship{
		ElementId:      "456",
		StartElementId: "1",
		EndElementId:   "2",
		Type:           "RELATES_TO",
		Props:          map[string]any{"uuid": "test-rel-uuid"},
	}

	tests := []struct {
		name   string
		input  any
		wantOK bool
	}{
		{"valid relationship", validRel, true},
		{"nil", nil, false},
		{"string", "not a rel", false},
		{"node (wrong type)", dbtype.Node{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsDBRelationship(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsDBRelationship() ok = %v, want %v", ok, tt.wantOK)
			}
			if tt.wantOK && got.ElementId != validRel.ElementId {
				t.Errorf("AsDBRelationship() ElementId = %s, want %s", got.ElementId, validRel.ElementId)
			}
		})
	}
}

func TestAsStringSlice(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  any
		wantOK bool
		wantN  int
	}{
		{"valid slice", []string{"a", "b", "c"}, true, 3},
		{"empty slice", []string{}, true, 0},
		{"nil", nil, false, 0},
		{"any slice", []any{"a", "b"}, false, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsStringSlice(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsStringSlice() ok = %v, want %v", ok, tt.wantOK)
			}
			if tt.wantOK && len(got) != tt.wantN {
				t.Errorf("AsStringSlice() len = %d, want %d", len(got), tt.wantN)
			}
		})
	}
}

func TestAsMap(t *testing.T) {
	t.Parallel()

	validMap := map[string]any{"key": "value", "count": int64(42)}

	tests := []struct {
		name   string
		input  any
		wantOK bool
	}{
		{"valid map", validMap, true},
		{"empty map", map[string]any{}, true},
		{"nil", nil, false},
		{"string map", map[string]string{"key": "value"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, ok := AsMap(tt.input)
			if ok != tt.wantOK {
				t.Errorf("AsMap() ok = %v, want %v", ok, tt.wantOK)
			}
			if tt.wantOK && tt.input != nil && len(got) != len(validMap) {
				// Only check length for non-empty valid maps
				if len(tt.input.(map[string]any)) > 0 {
					t.Errorf("AsMap() len = %d, want %d", len(got), len(validMap))
				}
			}
		})
	}
}

func TestMustString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     any
		field     string
		want      string
		wantError bool
	}{
		{"valid string", "hello", "name", "hello", false},
		{"nil", nil, "name", "", true},
		{"wrong type", 42, "name", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := MustString(tt.input, tt.field)
			if (err != nil) != tt.wantError {
				t.Errorf("MustString() error = %v, wantError %v", err, tt.wantError)
				return
			}
			if got != tt.want {
				t.Errorf("MustString() = %q, want %q", got, tt.want)
			}
			if tt.wantError {
				tce, ok := err.(*TypeConversionError)
				if !ok {
					t.Errorf("MustString() error type = %T, want *TypeConversionError", err)
				} else if tce.Field != tt.field {
					t.Errorf("MustString() error field = %q, want %q", tce.Field, tt.field)
				}
			}
		})
	}
}

func TestMustInt64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     any
		field     string
		want      int64
		wantError bool
	}{
		{"valid int64", int64(42), "count", 42, false},
		{"nil", nil, "count", 0, true},
		{"wrong type int", 42, "count", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := MustInt64(tt.input, tt.field)
			if (err != nil) != tt.wantError {
				t.Errorf("MustInt64() error = %v, wantError %v", err, tt.wantError)
				return
			}
			if got != tt.want {
				t.Errorf("MustInt64() = %d, want %d", got, tt.want)
			}
		})
	}
}
