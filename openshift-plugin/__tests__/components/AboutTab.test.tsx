/**
 * @jest-environment jsdom
 */
import * as React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AboutTab } from '../../src/core/components/AIModelSettings/tabs/AboutTab';

const mockFetchVersionInfo = jest.fn();
jest.mock('../../src/core/services/mcpClient', () => ({
  fetchVersionInfo: (...args: any[]) => mockFetchVersionInfo(...args),
}));

// In jsdom (no OpenShift Console context) getDeploymentMode() returns 'react-ui'
jest.mock('../../src/shared/config', () => ({
  getDeploymentMode: () => 'react-ui',
}));

// Default build version is 'dev' (not overwritten by Makefile in test)
jest.mock('../../src/generated/buildVersion', () => ({
  UI_BUILD_VERSION: 'dev',
}));

describe('AboutTab', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows spinner while loading', () => {
    mockFetchVersionInfo.mockReturnValue(new Promise(() => {}));
    render(<AboutTab />);

    expect(screen.getByLabelText('Loading version info')).toBeInTheDocument();
  });

  it('renders version data in description list', async () => {
    mockFetchVersionInfo.mockResolvedValue({ mcp_server: '1.5.0' });
    render(<AboutTab />);

    await waitFor(() => {
      expect(screen.getByText('MCP Server')).toBeInTheDocument();
      expect(screen.getByText('1.5.0')).toBeInTheDocument();
    });

    // UI label reflects react-ui mode in test environment
    expect(screen.getByText('React UI')).toBeInTheDocument();
    expect(screen.getByLabelText('Component version info')).toBeInTheDocument();
  });

  it('shows fallback message when fetch returns null', async () => {
    mockFetchVersionInfo.mockResolvedValue(null);
    render(<AboutTab />);

    await waitFor(() => {
      expect(screen.getByText('Unable to retrieve version')).toBeInTheDocument();
    });
  });

  it('shows dev (local) as default version on localhost', async () => {
    // jsdom defaults to localhost, so "dev" becomes "dev (local)"
    // MCP Server also returns "dev" when APP_VERSION is not set
    mockFetchVersionInfo.mockResolvedValue({ mcp_server: 'dev' });
    render(<AboutTab />);

    await waitFor(() => {
      const localLabels = screen.getAllByText('dev (local)');
      expect(localLabels).toHaveLength(2); // MCP Server + React UI
    });
  });
});
