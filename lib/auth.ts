import { cookies } from 'next/headers';
import bcrypt from 'bcryptjs';
import { getServerSupabase } from './supabase';
import { User, UserProfile, UserRole } from '@/types';

const SESSION_COOKIE = 'auth_session';
const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

interface SessionData {
  userId: number;
  email: string;
  name: string;
  role: UserRole;
}

/**
 * Get the current session from cookies
 */
export async function getSession(): Promise<SessionData | null> {
  const cookieStore = await cookies();
  const sessionCookie = cookieStore.get(SESSION_COOKIE);

  if (!sessionCookie) {
    return null;
  }

  try {
    return JSON.parse(sessionCookie.value);
  } catch {
    return null;
  }
}

/**
 * Set the session cookie
 */
export async function setSession(data: SessionData): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE, JSON.stringify(data), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: SESSION_MAX_AGE,
  });
}

/**
 * Clear the session cookie
 */
export async function clearSession(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete(SESSION_COOKIE);
}

/**
 * Get current user from session
 */
export async function getCurrentUser(): Promise<UserProfile | null> {
  const session = await getSession();
  if (!session) {
    return null;
  }

  return {
    id: session.userId,
    name: session.name,
    email: session.email,
    role: session.role,
  };
}

/**
 * Require authentication - throws if not logged in
 */
export async function requireAuth(): Promise<UserProfile> {
  const user = await getCurrentUser();
  if (!user) {
    throw new Error('Not authenticated');
  }
  return user;
}

/**
 * Login with email and password
 */
export async function login(email: string, password: string): Promise<UserProfile | null> {
  const supabase = getServerSupabase();

  const { data: user, error } = await supabase
    .from('users')
    .select('*')
    .eq('email', email.toLowerCase().trim())
    .single();

  if (error || !user) {
    return null;
  }

  const isValid = await bcrypt.compare(password, user.password_hash);
  if (!isValid) {
    return null;
  }

  // Set session
  await setSession({
    userId: user.id,
    email: user.email,
    name: user.name,
    role: user.role,
  });

  return {
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
  };
}

/**
 * Register a new user
 */
export async function register(
  name: string,
  email: string,
  password: string,
  role: UserRole = 'student'
): Promise<UserProfile | null> {
  const supabase = getServerSupabase();

  // Check if email already exists
  const { data: existing } = await supabase
    .from('users')
    .select('id')
    .eq('email', email.toLowerCase().trim())
    .single();

  if (existing) {
    throw new Error('Email already registered');
  }

  // Hash password
  const passwordHash = await bcrypt.hash(password, 10);

  // Insert user
  const { data: user, error } = await supabase
    .from('users')
    .insert({
      name: name.trim(),
      email: email.toLowerCase().trim(),
      password_hash: passwordHash,
      role,
    })
    .select()
    .single();

  if (error || !user) {
    console.error('Registration error:', error);
    return null;
  }

  // Set session
  await setSession({
    userId: user.id,
    email: user.email,
    name: user.name,
    role: user.role,
  });

  return {
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
  };
}

/**
 * Logout - clear session
 */
export async function logout(): Promise<void> {
  await clearSession();
}
