import { NextRequest, NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';
import { getServerSupabase } from '@/lib/supabase';

export async function POST(request: NextRequest) {
  try {
    const { email, name, newPassword } = await request.json();

    if (!email || !name || !newPassword) {
      return NextResponse.json(
        { error: 'Email, name, and new password are required' },
        { status: 400 }
      );
    }

    if (newPassword.length < 6) {
      return NextResponse.json(
        { error: 'Password must be at least 6 characters' },
        { status: 400 }
      );
    }

    const supabase = getServerSupabase();

    // Find user by email and verify name matches
    const { data: user, error: findError } = await supabase
      .from('users')
      .select('id, name')
      .eq('email', email.toLowerCase().trim())
      .single();

    if (findError || !user) {
      return NextResponse.json(
        { error: 'No account found with this email' },
        { status: 404 }
      );
    }

    // Verify name matches (case-insensitive)
    if (user.name.toLowerCase().trim() !== name.toLowerCase().trim()) {
      return NextResponse.json(
        { error: 'Name does not match our records' },
        { status: 400 }
      );
    }

    // Hash new password
    const passwordHash = await bcrypt.hash(newPassword, 10);

    // Update password
    const { error: updateError } = await supabase
      .from('users')
      .update({ password_hash: passwordHash })
      .eq('id', user.id);

    if (updateError) {
      console.error('Password update error:', updateError);
      return NextResponse.json(
        { error: 'Failed to update password' },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Reset password error:', error);
    return NextResponse.json(
      { error: 'Something went wrong' },
      { status: 500 }
    );
  }
}
