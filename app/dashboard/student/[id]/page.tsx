import { redirect, notFound } from 'next/navigation';
import { getCurrentUser } from '@/lib/auth';
import { getServerSupabase } from '@/lib/supabase';
import { scoreToLabel } from '@/services/analysis';
import Link from 'next/link';
import { DbSession, ProgressReport, UserProfile } from '@/types';

interface PageProps {
  params: Promise<{ id: string }>;
}

async function getStudentDetails(studentId: number) {
  const supabase = getServerSupabase();

  // Get student info
  const { data: student } = await supabase
    .from('users')
    .select('id, name, email, role')
    .eq('id', studentId)
    .single();

  if (!student || student.role !== 'student') {
    return null;
  }

  // Get sessions with analysis
  const { data: sessions } = await supabase
    .from('sessions')
    .select(`
      *,
      analysis:session_analysis(*)
    `)
    .eq('user_id', studentId)
    .order('date', { ascending: false });

  // Get progress reports
  const { data: reports } = await supabase
    .from('progress_reports')
    .select('*')
    .eq('user_id', studentId)
    .order('report_number', { ascending: true });

  return {
    student: student as UserProfile,
    sessions: (sessions || []) as DbSession[],
    reports: (reports || []) as ProgressReport[],
  };
}

export default async function StudentDetailPage({ params }: PageProps) {
  const user = await getCurrentUser();
  const { id } = await params;

  if (!user) {
    redirect('/login');
  }

  if (user.role !== 'teacher') {
    redirect('/dashboard');
  }

  const studentId = parseInt(id, 10);
  if (isNaN(studentId)) {
    notFound();
  }

  const data = await getStudentDetails(studentId);
  if (!data) {
    notFound();
  }

  const { student, sessions, reports } = data;
  const latestReport = reports[reports.length - 1];

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <Link
              href="/dashboard"
              className="text-sm text-blue-600 hover:underline mb-2 inline-block"
            >
              Back to Dashboard
            </Link>
            <h1 className="text-2xl font-bold text-slate-800">{student.name}</h1>
            <p className="text-slate-600">{student.email}</p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-slate-800">
              {sessions.length}
            </div>
            <div className="text-sm text-slate-500">sessions</div>
          </div>
        </div>

        {/* Progress Report */}
        {latestReport && (
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-lg font-semibold text-slate-800">
                Latest Progress Report
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

        {/* All Progress Reports */}
        {reports.length > 1 && (
          <div className="bg-white rounded-xl border border-slate-200 p-6">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">
              Progress History
            </h2>
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-200 text-left text-sm text-slate-500">
                  <th className="pb-3">Report</th>
                  <th className="pb-3">Sessions</th>
                  <th className="pb-3">Vocabulary</th>
                  <th className="pb-3">Phrasing</th>
                  <th className="pb-3">Structure</th>
                  <th className="pb-3">Overall</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r) => (
                  <tr key={r.id} className="border-b border-slate-100">
                    <td className="py-3 text-slate-500">{r.report_number}</td>
                    <td className="py-3 text-slate-500">
                      {r.sessions_from}–{r.sessions_to}
                    </td>
                    <td className="py-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          r.vocabulary_label === 'Fluent' ||
                          r.vocabulary_label === 'Mastery'
                            ? 'bg-green-100 text-green-700'
                            : r.vocabulary_label === 'Intermediate'
                            ? 'bg-blue-100 text-blue-700'
                            : r.vocabulary_label === 'Developing'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-orange-100 text-orange-700'
                        }`}
                      >
                        {r.vocabulary_score}
                      </span>
                    </td>
                    <td className="py-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          r.phrasing_label === 'Fluent' ||
                          r.phrasing_label === 'Mastery'
                            ? 'bg-green-100 text-green-700'
                            : r.phrasing_label === 'Intermediate'
                            ? 'bg-blue-100 text-blue-700'
                            : r.phrasing_label === 'Developing'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-orange-100 text-orange-700'
                        }`}
                      >
                        {r.phrasing_score}
                      </span>
                    </td>
                    <td className="py-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          r.structure_label === 'Fluent' ||
                          r.structure_label === 'Mastery'
                            ? 'bg-green-100 text-green-700'
                            : r.structure_label === 'Intermediate'
                            ? 'bg-blue-100 text-blue-700'
                            : r.structure_label === 'Developing'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-orange-100 text-orange-700'
                        }`}
                      >
                        {r.structure_score}
                      </span>
                    </td>
                    <td className="py-3">
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                          r.overall_label === 'Fluent' ||
                          r.overall_label === 'Mastery'
                            ? 'bg-green-100 text-green-700'
                            : r.overall_label === 'Intermediate'
                            ? 'bg-blue-100 text-blue-700'
                            : r.overall_label === 'Developing'
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-orange-100 text-orange-700'
                        }`}
                      >
                        {r.overall_score}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Session History */}
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">
            Session History
          </h2>

          {sessions.length === 0 ? (
            <p className="text-slate-500">No sessions yet.</p>
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
                {sessions.map((s) => {
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
