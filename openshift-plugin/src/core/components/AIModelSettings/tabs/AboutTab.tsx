import * as React from 'react';
import {
  DescriptionList,
  DescriptionListGroup,
  DescriptionListTerm,
  DescriptionListDescription,
  Spinner,
  Text,
  TextContent,
  TextVariants,
} from '@patternfly/react-core';
import { fetchVersionInfo, VersionInfo } from '../../../services/mcpClient';
import { getDeploymentMode } from '../../../../shared/config';
import { UI_BUILD_VERSION } from '../../../../generated/buildVersion';

const isLocalDev = typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

/**
 * Format a version string for display.
 * "dev" on localhost → "dev (local)" (running from source via local-dev.sh)
 * "dev" on OCP       → "dev"         (deployed but version not configured)
 */
function formatVersion(version: string): string {
  if (version === 'dev' && isLocalDev) {
    return 'dev (local)';
  }
  return version;
}

export const AboutTab: React.FC = () => {
  const [versionInfo, setVersionInfo] = React.useState<VersionInfo | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;

    async function loadVersion() {
      setLoading(true);
      setError(false);
      const info = await fetchVersionInfo();
      if (!cancelled) {
        if (info) {
          setVersionInfo(info);
        } else {
          setError(true);
        }
        setLoading(false);
      }
    }

    loadVersion();
    return () => { cancelled = true; };
  }, []);

  const mode = getDeploymentMode();
  const uiLabel = mode === 'plugin' ? 'Console Plugin' : 'React UI';

  if (loading) {
    return (
      <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'center', padding: '48px 0' }}>
        <Spinner size="lg" aria-label="Loading version info" />
      </div>
    );
  }

  return (
    <div style={{ marginTop: '16px' }}>
      <TextContent style={{ marginBottom: '16px' }}>
        <Text component={TextVariants.h4}>About</Text>
        <Text component={TextVariants.small} style={{ color: 'var(--pf-v5-global--Color--200)' }}>
          Deployed component versions.
        </Text>
      </TextContent>

      <DescriptionList isHorizontal aria-label="Component version info">
        <DescriptionListGroup>
          <DescriptionListTerm>MCP Server</DescriptionListTerm>
          <DescriptionListDescription>
            {error ? 'Unable to retrieve version' : formatVersion(versionInfo?.mcp_server ?? 'Unknown')}
          </DescriptionListDescription>
        </DescriptionListGroup>
        <DescriptionListGroup>
          <DescriptionListTerm>{uiLabel}</DescriptionListTerm>
          <DescriptionListDescription>{formatVersion(UI_BUILD_VERSION)}</DescriptionListDescription>
        </DescriptionListGroup>
      </DescriptionList>
    </div>
  );
};
