import { useState } from 'react';
import type { OpenApiGroup } from '../../types/api';
import { Icon } from '../Icon';
import { ApiEndpointRow } from './ApiEndpointRow';

export interface ApiGroupProps {
  group: OpenApiGroup;
  secured: boolean;
  token: string;
  baseUrl: string;
}

export function ApiGroup({ group, secured, token, baseUrl }: Readonly<ApiGroupProps>) {
  const [open, setOpen] = useState(true);
  return (
    <div className="swagger-group" data-testid={`api-group-${group.id}`}>
      <button className="swagger-group-head" onClick={() => setOpen((current) => !current)} aria-expanded={open}>
        <span className="swagger-group-name">{group.name}</span>
        <span className="swagger-group-desc">{group.desc}</span>
        <span className="swagger-group-count">{group.endpoints.length}</span>
        <span style={{ display: 'inline-flex', transform: open ? 'none' : 'rotate(-90deg)', transition: 'transform .15s' }}><Icon name="chevron-down" size={14} color="var(--color-text-tertiary)" /></span>
      </button>
      <div className="swagger-group-line" />
      {open && <div className="swagger-rows">{group.endpoints.map((endpoint, index) => <ApiEndpointRow key={`${endpoint.m}-${endpoint.p}-${index}`} ep={endpoint} secured={secured} token={token} baseUrl={baseUrl} />)}</div>}
    </div>
  );
}
