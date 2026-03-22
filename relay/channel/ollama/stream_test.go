package ollama

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestOllamaStreamHandlerClaudeRelayFormatReturnsClaudeSSE(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)

	info := &relaycommon.RelayInfo{
		IsStream:    true,
		RelayFormat: types.RelayFormatClaude,
		ChannelMeta: &relaycommon.ChannelMeta{
			UpstreamModelName: "llama3",
		},
		ClaudeConvertInfo: &relaycommon.ClaudeConvertInfo{
			LastMessagesType: relaycommon.LastMessageTypeNone,
		},
	}
	info.SetEstimatePromptTokens(11)

	body := strings.Join([]string{
		`{"model":"llama3","created_at":"2026-03-22T12:00:00Z","message":{"role":"assistant","content":"hello"},"done":false}`,
		`{"model":"llama3","created_at":"2026-03-22T12:00:01Z","done":true,"done_reason":"stop","prompt_eval_count":11,"eval_count":7}`,
	}, "\n")

	resp := &http.Response{
		Body: io.NopCloser(strings.NewReader(body)),
	}

	usage, newAPIError := ollamaStreamHandler(c, info, resp)
	require.Nil(t, newAPIError)
	require.NotNil(t, usage)
	require.Equal(t, 11, usage.PromptTokens)
	require.Equal(t, 7, usage.CompletionTokens)
	require.Equal(t, 18, usage.TotalTokens)

	output := recorder.Body.String()
	require.Contains(t, output, "event: message_start")
	require.Contains(t, output, "event: content_block_start")
	require.Contains(t, output, "event: content_block_delta")
	require.Contains(t, output, "event: content_block_stop")
	require.Contains(t, output, "event: message_delta")
	require.Contains(t, output, "event: message_stop")
	require.Contains(t, output, `"text":"hello"`)
	require.NotContains(t, output, "chat.completion.chunk")
	require.NotContains(t, output, "[DONE]")
}
