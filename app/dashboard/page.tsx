import { redirect } from 'next/navigation';
import { getCurrentUser } from '@/lib/auth';
import { getServerSupabase } from '@/lib/supabase';
import { scoreToLabel } from '@/services/analysis';
import Link from 'next/link';
import { LogoutButton } from '@/components/LogoutButton';
import { DbSession, ProgressReport, Topic, StudentSummary } from '@/types';

async function getStudentData(userId: number) {
  const supabase = getServerSupabase();

  // Get sessions with analysis
  const { data: sessions } = await supabase
    .from('sessions')
    .select(`
      *,
      analysis:session_analysis(*)
    `)
    .eq('user_id', userId)
    .order('date', { ascending: false });

  // Get progress reports
  const { data: reports } = await supabase
    .from('progress_reports')
    .select('*')
    .eq('user_id', userId)
    .order('report_number', { ascending: true });

  // Get all topics
  const { data: topics } = await supabase
    .from('eec_topics')
    .select('*')
    .order('topic_order', { ascending: true });

  return {
    sessions: (sessions || []) as DbSession[],
    reports: (reports || []) as ProgressReport[],
    topics: (topics || []) as Topic[],
  };
}

async function getTeacherData() {
  const supabase = getServerSupabase();

  // Get all students with their session counts
  const { data: students } = await supabase
    .from('users')
    .select('id, name, email')
    .eq('role', 'student')
    .order('name', { ascending: true });

  if (!students) return { students: [] };

  // Get session info for each student
  const studentSummaries: StudentSummary[] = [];

  for (const student of students) {
    const { data: sessions } = await supabase
      .from('sessions')
      .select(`
        id,
        date,
        analysis:session_analysis(overall_score)
      `)
      .eq('user_id', student.id)
      .order('date', { ascending: false })
      .limit(1);

    const latestSession = sessions?.[0];
    const { count } = await supabase
      .from('sessions')
      .select('*', { count: 'exact', head: true })
      .eq('user_id', student.id);

    studentSummaries.push({
      id: student.id,
      name: student.name,
      email: student.email,
      session_count: count || 0,
      latest_score: latestSession?.analysis?.[0]?.overall_score,
      latest_session_date: latestSession?.date,
    });
  }

  return { students: studentSummaries };
}

export default async function DashboardPage() {
  const user = await getCurrentUser();

  if (!user) {
    redirect('/login');
  }

  if (user.role === 'teacher') {
    // Teacher dashboard
    const { students } = await getTeacherData();

    return (
      <div className="min-h-screen bg-slate-50 p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-slate-800">
                Welcome, {user.name}
              </h1>
              <p className="text-slate-600">Teacher Dashboard</p>
            </div>
            <LogoutButton />
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">
              Students ({students.length})
            </h2>

            {students.length === 0 ? (
              <p className="text-slate-500">No students registered yet.</p>
            ) : (
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-sm text-slate-500">
                    <th className="pb-3">Name</th>
                    <th className="pb-3">Email</th>
                    <th className="pb-3">Sessions</th>
                    <th className="pb-3">Latest Score</th>
                    <th className="pb-3">Last Active</th>
                    <th className="pb-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {students.map((student) => (
                    <tr key={student.id} className="border-b border-slate-100">
                      <td className="py-3 font-medium">{student.name}</td>
                      <td className="py-3 text-slate-600">{student.email}</td>
                      <td className="py-3">{student.session_count}</td>
                      <td className="py-3">
                        {student.latest_score ? (
                          <span
                            className={`px-2 py-1 rounded-full text-xs font-medium ${
                              student.latest_score >= 7
                                ? 'bg-green-100 text-green-700'
                                : student.latest_score >= 5
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-orange-100 text-orange-700'
                            }`}
                          >
                            {student.latest_score}/10
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                      <td className="py-3 text-slate-600 text-sm">
                        {student.latest_session_date
                          ? new Date(student.latest_session_date).toLocaleDateString()
                          : '—'}
                      </td>
                      <td className="py-3">
                        <Link
                          href={`/dashboard/student/${student.id}`}
                          className="text-blue-600 hover:underline text-sm"
                        >
                          View details
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Student dashboard
  const { sessions, reports, topics } = await getStudentData(user.id);
  const latestReport = reports[reports.length - 1];

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-slate-800">
              Welcome back, {user.name}
            </h1>
            <p className="text-slate-600">
              {sessions.length} sessions completed
            </p>
          </div>
          <LogoutButton />
        </div>

        {/* Topic Selection */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">
            Choose a Topic
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {topics.map((topic) => (
              <Link
                key={topic.id}
                href={`/chat?style=clear&topicId=${topic.id}`}
                className="block p-4 border border-slate-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition"
              >
                <div className="font-medium text-slate-800">{topic.name}</div>
                <div className="text-sm text-slate-500">{topic.level}</div>
              </Link>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-slate-100">
            <Link
              href="/chat?style=casual"
              className="inline-flex items-center text-blue-600 hover:underline"
            >
              Or chat freely with Dora
            </Link>
          </div>
        </div>

        {/* Progress Report */}
        {latestReport && (
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-lg font-semibold text-slate-800">
                Progress Report
                <span className="text-sm font-normal text-slate-500 ml-2">
                  Sessions {latestReport.sessions_from}–{latestReport.sessions_to}
                </span>
              </h2>
            </div>

            <div className="space-y-4">
              {[
                {
                  name: 'Vocabulary',
                  score: latestReport.vocabulary_score,
                  label: latestReport.vocabulary_label,
                  desc: latestReport.vocabulary_description,
                },
                {
                  name: 'Phrasing',
                  score: latestReport.phrasing_score,
                  label: latestReport.phrasing_label,
                  desc: latestReport.phrasing_description,
                },
                {
                  name: 'Structure',
                  score: latestReport.structure_score,
                  label: latestReport.structure_label,
                  desc: latestReport.structure_description,
                },
              ].map((cat) => (
                <div key={cat.name}>
                  <div className="flex items-center gap-3 mb-1">
                    <span className="font-medium text-slate-700">{cat.name}</span>
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                        cat.label === 'Fluent' || cat.label === 'Mastery'
                          ? 'bg-green-100 text-green-700'
                          : cat.label === 'Intermediate'
                          ? 'bg-blue-100 text-blue-700'
                          : cat.label === 'Developing'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-orange-100 text-orange-700'
                      }`}
                    >
                      {cat.label}
                    </span>
                    <span className="text-sm text-slate-500">{cat.score}/10</span>
                  </div>
                  <p className="text-sm text-slate-600">{cat.desc}</p>
                </div>
              ))}

              <div className="pt-4 border-t border-slate-100">
                <div className="flex items-center gap-3 mb-1">
                  <span className="font-semibold text-slate-800">Overall</span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      latestReport.overall_label === 'Fluent' ||
                      latestReport.overall_label === 'Mastery'
                        ? 'bg-green-100 text-green-700'
                        : latestReport.overall_label === 'Intermediate'
                        ? 'bg-blue-100 text-blue-700'
                        : latestReport.overall_label === 'Developing'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-orange-100 text-orange-700'
                    }`}
                  >
                    {latestReport.overall_label}
                  </span>
                  <span className="text-sm text-slate-500">
                    {latestReport.overall_score}/10
                  </span>
                </div>
                <p className="text-sm text-slate-600">
                  {latestReport.improvement_description}
                </p>
              </div>
            </div>
          </div>
        )}

        {!latestReport && sessions.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-6 text-center text-slate-500">
            Your first progress report will be generated after 5 sessions.
            <br />
            {5 - (sessions.length % 5)} more to go!
          </div>
        )}

        {/* Session History */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">
            Session History
          </h2>

          {sessions.length === 0 ? (
            <p className="text-slate-500">
              No sessions yet. Start your first conversation!
            </p>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-200 text-left text-sm text-slate-500">
                  <th className="pb-3">#</th>
                  <th className="pb-3">Topic</th>
                  <th className="pb-3">Date</th>
                  <th className="pb-3">Score</th>
                  <th className="pb-3">Suggestion</th>
                </tr>
              </thead>
              <tbody>
                {sessions.slice(0, 10).map((s) => {
                  const analysis = Array.isArray(s.analysis)
                    ? s.analysis[0]
                    : s.analysis;
                  return (
                    <tr key={s.id} className="border-b border-slate-100">
                      <td className="py-3 text-slate-500">{s.session_number}</td>
                      <td className="py-3">{s.topic}</td>
                      <td className="py-3 text-slate-500 text-sm">
                        {new Date(s.date).toLocaleDateString()}
                      </td>
                      <td className="py-3">
                        {analysis?.overall_score ? (
                          <span
                            className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                              analysis.overall_score >= 7
                                ? 'bg-green-100 text-green-700'
                                : analysis.overall_score >= 5
                                ? 'bg-yellow-100 text-yellow-700'
                                : 'bg-orange-100 text-orange-700'
                            }`}
                          >
                            {scoreToLabel(analysis.overall_score)}
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-3 text-sm text-slate-600 max-w-xs truncate">
                        {analysis?.suggestion || '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
