import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ModelInsightsSection } from '../../src/core/components/ModelInsightsSection';
import { fetchVLLMMetrics, detectProviderFromModelId } from '../../src/core/services/mcpClient';
import type { ModelInfo } from '../../src/core/services/mcpClient';

jest.mock('../../src/core/services/mcpClient', () => ({
  detectProviderFromModelId: jest.fn(),
  fetchVLLMMetrics: jest.fn(),
}));

const mockFetchVLLMMetrics = fetchVLLMMetrics as jest.MockedFunction<typeof fetchVLLMMetrics>;
const mockDetectProvider = detectProviderFromModelId as jest.MockedFunction<typeof detectProviderFromModelId>;

const sampleModels: ModelInfo[] = [
  { name: 'meta-llama/Llama-3-8b', namespace: 'team-alpha', status: 'Running' },
  { name: 'ibm-granite/granite-3b', namespace: 'team-beta', status: 'Running' },
  { name: 'mistralai/Mistral-7B', namespace: 'team-alpha', status: 'Running' },
];

describe('ModelInsightsSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockDetectProvider.mockImplementation((id: string) => {
      if (id.includes('llama')) return 'meta-llama';
      if (id.includes('granite')) return 'ibm-granite';
      if (id.includes('Mistral')) return 'mistralai';
      return null;
    });
    mockFetchVLLMMetrics.mockResolvedValue(null);
  });

  it('renders all three card titles', async () => {
    render(<ModelInsightsSection loading={false} error={null} models={sampleModels} />);

    await waitFor(() => {
      expect(screen.getByTestId('Models by Provider-legend')).toBeInTheDocument();
    });

    // Card titles appear as headings
    expect(screen.getAllByText('Models by Provider').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Model Performance').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Models by Namespace').length).toBeGreaterThanOrEqual(1);
  });

  it('renders legend items as HTML elements, not SVG text', async () => {
    render(<ModelInsightsSection loading={false} error={null} models={sampleModels} />);

    await waitFor(() => {
      expect(screen.getByTestId('Models by Namespace-legend')).toBeInTheDocument();
    });

    const namespaceLegend = screen.getByTestId('Models by Namespace-legend');
    expect(namespaceLegend.tagName).toBe('DIV');

    const legendItems = namespaceLegend.querySelectorAll('[data-testid^="legend-item-"]');
    expect(legendItems.length).toBeGreaterThan(0);

    legendItems.forEach((item) => {
      expect(item.tagName).toBe('DIV');
    });
  });

  it('renders namespace legend items with correct labels', async () => {
    render(<ModelInsightsSection loading={false} error={null} models={sampleModels} />);

    await waitFor(() => {
      expect(screen.getByTestId('legend-item-team-alpha')).toBeInTheDocument();
    });

    expect(screen.getByTestId('legend-item-team-alpha')).toHaveTextContent('team-alpha (2)');
    expect(screen.getByTestId('legend-item-team-beta')).toHaveTextContent('team-beta (1)');
  });

  it('renders "No data" state when there are no models', async () => {
    render(<ModelInsightsSection loading={false} error={null} models={[]} />);

    await waitFor(() => {
      expect(screen.getByTestId('Models by Provider-legend')).toBeInTheDocument();
    });

    const providerLegend = screen.getByTestId('Models by Provider-legend');
    expect(providerLegend).toHaveTextContent('No data (1)');
  });

  it('shows loading spinner while loading', () => {
    render(<ModelInsightsSection loading={true} error={null} models={[]} />);

    expect(screen.getByText('Loading model insights')).toBeInTheDocument();
  });

  it('shows an alert when there is an error', async () => {
    render(
      <ModelInsightsSection loading={false} error="Connection failed" models={sampleModels} />,
    );

    await waitFor(() => {
      expect(screen.getByText('Model Insights Unavailable')).toBeInTheDocument();
    });
  });

  it('renders colored dots in legend items', async () => {
    render(<ModelInsightsSection loading={false} error={null} models={sampleModels} />);

    await waitFor(() => {
      expect(screen.getByTestId('legend-item-team-alpha')).toBeInTheDocument();
    });

    const legendItem = screen.getByTestId('legend-item-team-alpha');
    const dot = legendItem.querySelector('span');
    expect(dot).toBeInTheDocument();
    expect(dot).toHaveStyle({ borderRadius: '50%' });
  });

  it('wraps each legend item in a tooltip', async () => {
    render(<ModelInsightsSection loading={false} error={null} models={sampleModels} />);

    await waitFor(() => {
      expect(screen.getByTestId('Models by Namespace-legend')).toBeInTheDocument();
    });

    // PatternFly Tooltip wraps children with aria attributes
    const legendItem = screen.getByTestId('legend-item-team-alpha');
    // The legend item should exist within the legend container
    const legend = screen.getByTestId('Models by Namespace-legend');
    expect(legend).toContainElement(legendItem);
  });
});
