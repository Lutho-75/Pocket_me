-- Pocket Me — business profile storage
-- ---------------------------------------------------------------------
-- Run this once in your Supabase project:
--   Dashboard -> SQL Editor -> New query -> paste -> Run
--
-- It creates the table the AI consultant reads the user's business
-- profile from, and locks it down with row-level security so each user
-- can only ever see and edit their own row.

create table if not exists public.profiles (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  data       jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

-- Recreate policies idempotently so re-running this script is safe.
drop policy if exists "Users can read their own profile"   on public.profiles;
drop policy if exists "Users can insert their own profile" on public.profiles;
drop policy if exists "Users can update their own profile" on public.profiles;
drop policy if exists "Users can delete their own profile" on public.profiles;

create policy "Users can read their own profile"
  on public.profiles for select
  using (auth.uid() = user_id);

create policy "Users can insert their own profile"
  on public.profiles for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own profile"
  on public.profiles for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Users can delete their own profile"
  on public.profiles for delete
  using (auth.uid() = user_id);
