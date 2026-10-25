// TanStack Query для ролей, учётных записей и журнала действий (docs/ROLES.md, «API»)

import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import { fallbackRolesCatalog, type AuditQuery, type RolesCatalog, type UserCreate, type UserPatch } from '../domain/auth';
import { useAuth, usePermission } from '../state/auth';
import { api } from './index';
import { ApiError } from './types';

export const authKeys = {
  roles: ['auth', 'roles'] as const,
  users: ['users'] as const,
  auditAll: ['audit'] as const,
  audit: (q: AuditQuery) => ['audit', q] as const,
};

/** 4xx не повторяем: ответ не изменится, а 403 породил бы лишние уведомления */
const retryUnlessClientError = (count: number, error: unknown): boolean =>
  !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 2;

/**
 * Каталог ролей с сервера (GET /api/auth/roles, нужен только вход). Пока он не
 * ответил или при выключенном входе — матрица по docs/ROLES.md.
 */
export function useRolesCatalog(): { catalog: RolesCatalog; fromServer: boolean; isLoading: boolean } {
  const enabled = useAuth((s) => s.status === 'authenticated' && Boolean(s.config?.auth_enabled));
  const query = useQuery({
    queryKey: authKeys.roles,
    queryFn: () => api().auth.roles(),
    enabled,
    staleTime: 5 * 60_000,
    retry: retryUnlessClientError,
  });
  const fallback = useMemo(fallbackRolesCatalog, []);
  const fromServer = Boolean(query.data && query.data.roles.length > 0);
  return { catalog: fromServer && query.data ? query.data : fallback, fromServer, isLoading: enabled && query.isLoading };
}

export function useUsers() {
  const allowed = usePermission('users.manage');
  return useQuery({
    queryKey: authKeys.users,
    queryFn: () => api().users.list(),
    enabled: allowed,
    staleTime: 15_000,
    retry: retryUnlessClientError,
  });
}

/** После любой правки учётных записей освежаем список и журнал действий */
function useInvalidateUsers() {
  const qc = useQueryClient();
  return () => {
    void qc.invalidateQueries({ queryKey: authKeys.users });
    void qc.invalidateQueries({ queryKey: authKeys.auditAll });
  };
}

export function useCreateUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({
    mutationFn: (user: UserCreate) => api().users.create(user),
    onSuccess: invalidate,
  });
}

export function useUpdateUser() {
  const invalidate = useInvalidateUsers();
  return useMutation({
    mutationFn: ({ username, patch }: { username: string; patch: UserPatch }) => api().users.update(username, patch),
    onSuccess: invalidate,
  });
}

export function useAudit(query: AuditQuery) {
  const allowed = usePermission('audit.view');
  return useQuery({
    queryKey: authKeys.audit(query),
    queryFn: ({ signal }) => api().audit.list(query, signal),
    enabled: allowed,
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    refetchInterval: 30_000,
    retry: retryUnlessClientError,
  });
}
