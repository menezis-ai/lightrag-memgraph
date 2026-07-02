import { GRAPH_TYPE_LABEL, type GraphEntity, type GraphEntityType } from '../../types/graph';

export const TYPE_KEYS = Object.keys(GRAPH_TYPE_LABEL) as readonly GraphEntityType[];

export const entityRadius = (entity: GraphEntity): number =>
  8 + Math.min(18, Math.sqrt(entity.mentions) * 0.9);
