import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Lazy-loaded clients to avoid build-time errors
let clientInstance: SupabaseClient | null = null;
let serverInstance: SupabaseClient | null = null;

// Client-side Supabase client (lazy-loaded)
export const getSupabase = () => {
  if (!clientInstance) {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseAnonKey) {
      throw new Error('Supabase environment variables not configured');
    }

    clientInstance = createClient(supabaseUrl, supabaseAnonKey);
  }
  return clientInstance;
};

// Server-side Supabase client (with service role key for API routes)
export const getServerSupabase = () => {
  if (!serverInstance) {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

    if (!supabaseUrl) {
      throw new Error('NEXT_PUBLIC_SUPABASE_URL not configured');
    }

    // Use service role key if available, otherwise fall back to anon key
    const key = serviceRoleKey || supabaseAnonKey;
    if (!key) {
      throw new Error('No Supabase key configured');
    }

    serverInstance = createClient(supabaseUrl, key);
  }
  return serverInstance;
};
