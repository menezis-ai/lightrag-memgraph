import {
  useApproveTag,
  useDeleteTag,
  useDeprecateTag,
  useEditTag,
  useReactivateTag,
  useRejectTag,
  useRequestTag,
  useSuggestTagEdit,
  useUpdateTagSynonyms,
} from '../api/queries';
import type { TagActionCommit } from '../components/TagActionModal';
import type { TagApproveAction } from '../components/TagsTab';
import type { Toast } from '../types/toast';

interface UseTagActionsOptions {
  currentActor: string;
  pushToast: (toast: Omit<Toast, 'id'>) => void;
}

export function useTagActions({
  currentActor,
  pushToast,
}: UseTagActionsOptions) {
  // Tag mutations call the backend through TanStack Query. Each mutation
  // invalidates ['tags']+['activity']+['notifications'] on success so the
  // operator sees the new state + audit event + notification on refetch.
  const requestTag = useRequestTag();
  const approveTag = useApproveTag();
  const rejectTag = useRejectTag();
  const editTag = useEditTag();
  const suggestTagEdit = useSuggestTagEdit();
  const deprecateTag = useDeprecateTag();
  const reactivateTag = useReactivateTag();
  const updateSynonyms = useUpdateTagSynonyms();
  const deleteTag = useDeleteTag();

  const onTagApprove = async (action: TagApproveAction) => {
    try {
      await approveTag.mutateAsync({
        name: action.tag.tag,
        actor: currentActor,
      });
      pushToast({
        kind: 'done',
        title: 'Tag',
        tagname: action.tag.tag,
        titleSuffix: 'approved',
        sub: 'Added to tag catalog · Tier 3',
      });
    } catch (err) {
      pushToast({
        kind: 'error',
        title: 'Tag approval failed',
        tagname: action.tag.tag,
        sub: err instanceof Error ? err.message : 'Mutation rejected',
      });
    }
  };

  const commitTagMutation = (
    run: (callbacks: {
      onSuccess: () => void;
      onError: (err: unknown) => void;
    }) => void,
    toast: Omit<Toast, 'id'>,
    failureTitle: string,
  ) => {
    run({
      onSuccess: () => pushToast(toast),
      onError: (err) =>
        pushToast({
          kind: 'error',
          title: failureTitle,
          tagname: toast.tagname,
          sub: err instanceof Error ? err.message : 'Mutation rejected',
        }),
    });
  };

  const onTagCommit = (commit: TagActionCommit) => {
    const tagname = commit.tag?.tag ?? commit.name ?? '';
    const actor = currentActor;
    const verbMap: Record<TagActionCommit['kind'], string> = {
      edit: 'updated',
      suggest: 'edit suggested',
      synonyms: 'synonyms updated',
      deprecate: 'deprecated',
      reactivate: 'reactivated',
      delete:
        commit.migrate?.strategy === 'migrate'
          ? `migrated to ${commit.migrate.to ?? ''}`
          : 'deleted (docs untagged)',
      reject: 'rejected',
      'edit-approve': 'approved (edited)',
      request: 'requested for review',
    };
    const successToast: Omit<Toast, 'id'> = {
      kind: 'done',
      title: 'Tag',
      tagname,
      titleSuffix: verbMap[commit.kind],
      sub: commit.reason ?? '',
    };
    const failureTitle = `Tag ${commit.kind} failed`;

    switch (commit.kind) {
      case 'edit':
        commitTagMutation(
          (cb) =>
            editTag.mutate(
              {
                name: tagname,
                tag: commit.name,
                def: commit.def,
                long_description: commit.longDescription,
                category: commit.category,
                actor,
              },
              cb,
            ),
          successToast,
          failureTitle,
        );
        break;
      case 'suggest':
        if (commit.tag) {
          commitTagMutation(
            (cb) =>
              suggestTagEdit.mutate(
                {
                  name: tagname,
                  def: commit.def,
                  long_description: commit.longDescription,
                  category: commit.category,
                  aliases: commit.aliases ?? commit.tag!.aliases,
                  justification: commit.justification,
                  actor,
                },
                cb,
              ),
            successToast,
            failureTitle,
          );
        }
        break;
      case 'synonyms':
        if (commit.tag) {
          commitTagMutation(
            (cb) =>
              updateSynonyms.mutate(
                {
                  name: tagname,
                  aliases: commit.aliases ?? commit.tag!.aliases,
                  actor,
                },
                cb,
              ),
            successToast,
            failureTitle,
          );
        }
        break;
      case 'deprecate':
        commitTagMutation(
          (cb) =>
            deprecateTag.mutate({ name: tagname, actor, reason: commit.reason }, cb),
          successToast,
          failureTitle,
        );
        break;
      case 'reactivate':
        commitTagMutation(
          (cb) => reactivateTag.mutate({ name: tagname, actor }, cb),
          successToast,
          failureTitle,
        );
        break;
      case 'delete':
        commitTagMutation(
          (cb) =>
            deleteTag.mutate(
              {
                name: tagname,
                strategy: commit.migrate?.strategy ?? 'untag',
                to: commit.migrate?.to,
                actor,
              },
              cb,
            ),
          successToast,
          failureTitle,
        );
        break;
      case 'reject':
        commitTagMutation(
          (cb) =>
            rejectTag.mutate(
              {
                name: tagname,
                reason: commit.reason || 'rejected',
                actor,
              },
              cb,
            ),
          successToast,
          failureTitle,
        );
        break;
      case 'edit-approve':
        void (async () => {
          try {
            if (
              commit.name ||
              commit.def ||
              commit.longDescription ||
              commit.category
            ) {
              await editTag.mutateAsync({
                name: tagname,
                tag: commit.name,
                def: commit.def,
                long_description: commit.longDescription,
                category: commit.category,
                actor,
              });
            }
            // After an edit that renames the tag, the old name no longer
            // exists — approve the NEW name. Trim to match the backend, which
            // stores the trimmed rename target (a raw name would 404 on approve).
            await approveTag.mutateAsync({
              name: commit.name?.trim() || tagname,
              actor,
            });
            pushToast(successToast);
          } catch (err) {
            pushToast({
              kind: 'error',
              title: failureTitle,
              tagname: successToast.tagname,
              sub: err instanceof Error ? err.message : 'Mutation rejected',
            });
          }
        })();
        break;
      case 'request':
        if (commit.name) {
          commitTagMutation(
            (cb) =>
              requestTag.mutate(
                {
                  tag: commit.name!,
                  def: commit.def ?? '',
                  long_description: commit.longDescription,
                  category: commit.category ?? 'infra',
                  aliases: commit.aliases ?? [],
                  justification: commit.justification,
                  actor,
                },
                cb,
              ),
            successToast,
            failureTitle,
          );
        }
        break;
    }
  };

  return { onTagApprove, onTagCommit };
}
