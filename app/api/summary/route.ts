import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { buildReview } from '@/services/review';
import { analyzeSession, analyzeProgress } from '@/services/analysis';
import { logTopicCompletion } from '@/lib/db';
import { getSession as getAuthSession } from '@/lib/auth';
import { getServerSupabase } from '@/lib/supabase';
import { PracticeTurn, Topic } from '@/types';

async function getSession() {
  const cookieStore = await cookies();
  const sessionCookie = cookieStore.get('session');
  if (sessionCookie) {
    try {
      return JSON.parse(sessionCookie.value);
    } catch {
      return null;
    }
  }
  return null;
}

export async function POST() {
  try {
    const session = (await getSession()) || {};
    const authSession = await getAuthSession();
    const userId = authSession?.userId || null;

    const style: string = session.style || 'casual';
    const turns: PracticeTurn[] = session.turns || [];
    const topic: Topic | null = session.topic || null;
    const topicId: number | null = session.topicId || null;
    const topicName = topic?.name || '';
    const userName: string = session.userName || 'Ray';

    // Build the review markdown
    const summary = buildReview(turns, style, topicName);

    // Log topic completion (for topic progression)
    if (style === 'clear' && topicId) {
      try {
        await logTopicCompletion(topicId, userName);
      } catch (error) {
        console.error('Topic completion log error:', error);
      }
    }

    // Save session and analysis for authenticated users
    let analysisData = null;
    let progressReportData = null;

    if (userId && turns.length > 0) {
      try {
        const supabase = getServerSupabase();

        // Get current session count
        const { count: sessionCount } = await supabase
          .from('sessions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', userId);

        const sessionNumber = (sessionCount || 0) + 1;

        // Create session record
        const { data: dbSession, error: sessionError } = await supabase
          .from('sessions')
          .insert({
            user_id: userId,
            topic: topicName || 'Free conversation',
            topic_id: topicId,
            session_number: sessionNumber,
          })
          .select()
          .single();

        if (sessionError) {
          console.error('Session save error:', sessionError);
        } else if (dbSession) {
          // Save turns
          const turnRecords = turns.map((t, i) => ({
            session_id: dbSession.id,
            turn_number: i + 1,
            app_question: t.morgan || '',
            student_speech: t.student || '',
            fluency_comment: t.corrected || t.reply || '',
          }));

          await supabase.from('turns').insert(turnRecords);

          // Analyze session
          const turnsForAnalysis = turns.map((t) => ({
            app_question: t.morgan || '',
            student_speech: t.student || '',
            fluency_comment: t.corrected || t.reply || '',
          }));

          analysisData = await analyzeSession(turnsForAnalysis);

          // Save analysis
          await supabase.from('session_analysis').insert({
            session_id: dbSession.id,
            ...analysisData,
          });

          // Generate progress report every 5 sessions
          if (sessionNumber % 5 === 0) {
            const { data: last5Sessions } = await supabase
              .from('sessions')
              .select(`
                topic,
                analysis:session_analysis(
                  vocabulary_score,
                  vocabulary_note,
                  phrasing_score,
                  phrasing_note,
                  structure_score,
                  structure_note,
                  overall_score,
                  overall_note
                )
              `)
              .eq('user_id', userId)
              .order('session_number', { ascending: false })
              .limit(5);

            if (last5Sessions && last5Sessions.length > 0) {
              const sessionsData = last5Sessions
                .filter((s) => s.analysis && s.analysis.length > 0)
                .map((s) => ({
                  topic: s.topic,
                  vocabulary_score: s.analysis[0].vocabulary_score,
                  vocabulary_note: s.analysis[0].vocabulary_note,
                  phrasing_score: s.analysis[0].phrasing_score,
                  phrasing_note: s.analysis[0].phrasing_note,
                  structure_score: s.analysis[0].structure_score,
                  structure_note: s.analysis[0].structure_note,
                  overall_score: s.analysis[0].overall_score,
                  overall_note: s.analysis[0].overall_note,
                }))
                .reverse();

              if (sessionsData.length > 0) {
                progressReportData = await analyzeProgress(sessionsData);

                // Get report number
                const { count: reportCount } = await supabase
                  .from('progress_reports')
                  .select('*', { count: 'exact', head: true })
                  .eq('user_id', userId);

                const reportNumber = (reportCount || 0) + 1;

                await supabase.from('progress_reports').insert({
                  user_id: userId,
                  report_number: reportNumber,
                  sessions_from: sessionNumber - 4,
                  sessions_to: sessionNumber,
                  ...progressReportData,
                });
              }
            }
          }
        }
      } catch (error) {
        console.error('Session tracking error:', error);
      }
    }

    // Return practice turns for the replay feature
    const practiceTurns: PracticeTurn[] = turns.map((t) => ({
      morgan: t.morgan || '',
      student: t.student || '',
      corrected: t.corrected || '',
      reply: t.reply || '',
    }));

    return NextResponse.json({
      summary,
      practice_turns: practiceTurns,
      analysis: analysisData,
      progress_report: progressReportData,
    });
  } catch (error) {
    console.error('Summary error:', error);
    return NextResponse.json(
      {
        summary:
          '**Sorry, the review could not be generated this time.** Please try again, or start a new one.',
        practice_turns: [],
      },
      { status: 200 }
    );
  }
}
