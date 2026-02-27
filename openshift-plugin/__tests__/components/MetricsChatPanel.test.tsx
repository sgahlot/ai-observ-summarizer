import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MetricsChatPanel } from '../../src/core/components/MetricsChatPanel';
import * as mcpClient from '../../src/core/services/mcpClient';

// Mock services
jest.mock('../../src/core/services/mcpClient');

// Mock ReactMarkdown (ESM module)
jest.mock('react-markdown', () => ({
  __esModule: true,
  default: ({ children }: any) => <div>{children}</div>,
}));

jest.mock('../../src/core/components/ConfigurationRequiredAlert', () => ({
  ConfigurationRequiredAlert: ({ onClose }: any) => (
    <div data-testid="config-alert">
      <button onClick={onClose}>Close</button>
    </div>
  ),
}));

describe('MetricsChatPanel', () => {
  const mockGetSessionConfig = mcpClient.getSessionConfig as jest.MockedFunction<typeof mcpClient.getSessionConfig>;
  const mockChatOpenShift = mcpClient.chatOpenShift as jest.MockedFunction<typeof mcpClient.chatOpenShift>;

  const defaultProps = {
    pageType: 'openshift' as const,
    scope: 'cluster_wide',
    namespace: undefined,
    category: 'Fleet Overview',
    timeRange: '1h',
    isOpen: true,
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    mockGetSessionConfig.mockReturnValue({
      ai_model: 'test-model',
      api_key: 'test-key',
    });
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe('Focus Management', () => {
    it('should focus input when panel opens', () => {
      render(<MetricsChatPanel {...defaultProps} />);

      // Advance timer for the 100ms setTimeout in the focus useEffect
      act(() => {
        jest.advanceTimersByTime(100);
      });

      const input = screen.getByLabelText('Type your question');
      expect(document.activeElement).toBe(input);
    });

    it('should not render when panel is closed', () => {
      const { container } = render(<MetricsChatPanel {...defaultProps} isOpen={false} />);
      expect(container.innerHTML).toBe('');
    });

    it('should focus input after successful response', async () => {
      mockChatOpenShift.mockResolvedValue({
        response: 'AI response',
        promql: undefined,
      });

      render(<MetricsChatPanel {...defaultProps} />);

      const input = screen.getByLabelText('Type your question') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'Test question' } });
      fireEvent.keyPress(input, { key: 'Enter', charCode: 13 });

      await waitFor(() => {
        expect(mockChatOpenShift).toHaveBeenCalled();
      });

      await waitFor(() => {
        expect(document.activeElement).toBe(input);
      });
    });

    it('should focus input after error response', async () => {
      mockChatOpenShift.mockRejectedValue(new Error('Network error'));

      render(<MetricsChatPanel {...defaultProps} />);

      const input = screen.getByLabelText('Type your question') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'Test question' } });
      fireEvent.keyPress(input, { key: 'Enter', charCode: 13 });

      await waitFor(() => {
        expect(document.activeElement).toBe(input);
      });
    });
  });
});
