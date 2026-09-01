/**
 * ApiTab — explorer-wide filtering and authorization state.
 * Endpoint execution and request projection are decomposed under ApiExplorer/.
 */
/* eslint-disable react-refresh/only-export-components -- compatibility re-exports keep the established ApiTab helper contract. */
import { useState } from 'react';
import type { OpenApiGroup } from '../types/api';
import { ApiGroup } from './ApiExplorer/ApiGroup';
import { AuthorizeDialog } from './ApiExplorer/AuthorizeDialog';
import { Icon } from './Icon';

export { curlFor, paramKey, requestBodyFor, resolveRequestTarget, responsesFor, type ResolvedTarget } from './ApiExplorer/apiRequest';

export interface ApiTabProps {
  apiVersion: string;
  groups: readonly OpenApiGroup[];
  baseUrl: string;
}

export function ApiTab({ apiVersion, groups, baseUrl }: Readonly<ApiTabProps>) {
  const [filter, setFilter] = useState('');
  const [authOpen, setAuthOpen] = useState(false);
  const [token, setToken] = useState('');
  const normalizedFilter = filter.trim().toLowerCase();
  const filteredGroups = groups
    .map((group) => ({
      ...group,
      endpoints: group.endpoints.filter((endpoint) =>
        !normalizedFilter ||
        endpoint.p.toLowerCase().includes(normalizedFilter) ||
        endpoint.s.toLowerCase().includes(normalizedFilter) ||
        endpoint.m.toLowerCase() === normalizedFilter,
      ),
    }))
    .filter((group) => group.endpoints.length > 0);

  return (
    <div className="swagger">
      <div className="swagger-topbar">
        <div className="swagger-title">
          <span className="swagger-title-main">Twin KMS API</span>
          <span className="swagger-version">{apiVersion}</span>
          <span className="swagger-oas">OAS 3.1</span>
        </div>
        <div className="swagger-meta">
          <code>/openapi.json</code><span className="swagger-sep">·</span>
          <span>Documents, folders, tags, knowledge graph and grounded retrieval</span>
        </div>
        <div className="swagger-servers">
          <span className="swagger-server-current"><Icon name="world" size={12} /> {baseUrl || '(no server)'}</span>
          <button className={'swagger-auth' + (token ? ' is-on' : '')} onClick={() => setAuthOpen(true)}>
            <Icon name={token ? 'circle-check' : 'lock'} size={12} color={token ? 'var(--twin-green-700)' : 'var(--color-text-secondary)'} />
            {token ? 'Authorized' : 'Authorize'}
          </button>
        </div>
        <div className="swagger-filter">
          <Icon name="search" size={13} color="var(--color-text-tertiary)" />
          <input type="text" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter by path, summary or method (GET, POST…)" aria-label="Filter endpoints" />
          {filter && <button className="swagger-filter-clear" onClick={() => setFilter('')} aria-label="Clear filter"><Icon name="x" size={12} color="var(--color-text-tertiary)" /></button>}
        </div>
      </div>

      <div className="swagger-groups">
        {filteredGroups.map((group) => <ApiGroup key={group.id} group={group} secured={group.id !== 'default'} token={token} baseUrl={baseUrl} />)}
        {filteredGroups.length === 0 && <div className="empty-state" style={{ padding: 60 }}><div className="title">No endpoints match &quot;{filter}&quot;</div></div>}
      </div>

      {authOpen && <AuthorizeDialog token={token} onSave={(nextToken) => { setToken(nextToken); setAuthOpen(false); }} onLogout={() => { setToken(''); setAuthOpen(false); }} onClose={() => setAuthOpen(false)} />}
    </div>
  );
}
