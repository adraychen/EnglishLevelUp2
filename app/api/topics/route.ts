import { NextResponse } from 'next/server';
import { getServerSupabase } from '@/lib/supabase';

export async function GET() {
  try {
    const supabase = getServerSupabase();

    const { data: topics, error } = await supabase
      .from('eec_topics')
      .select('*')
      .order('topic_order', { ascending: true });

    if (error) {
      console.error('Error fetching topics:', error);
      return NextResponse.json({ error: 'Failed to fetch topics' }, { status: 500 });
    }

    return NextResponse.json({ topics: topics || [] });
  } catch (error) {
    console.error('Topics error:', error);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
