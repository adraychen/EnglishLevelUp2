import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6">
      <div className="max-w-md w-full text-center space-y-8">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold text-slate-800">
            English Conversation Coach
          </h1>
          <p className="text-slate-600">
            Practice English naturally with AI coaches who help you improve
            through conversation.
          </p>
        </div>

        <div className="grid gap-4">
          <Link
            href="/chat?style=casual"
            className="block p-6 bg-white border border-slate-200 rounded-xl shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="text-2xl mb-2">Dora</div>
            <div className="text-lg font-semibold text-slate-800 mb-1">
              Casual Conversation
            </div>
            <p className="text-sm text-slate-600">
              Free, natural chat on any subject. Great for intermediate to
              advanced learners.
            </p>
          </Link>

          <Link
            href="/chat?style=clear"
            className="block p-6 bg-white border border-slate-200 rounded-xl shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="text-2xl mb-2">Morgan</div>
            <div className="text-lg font-semibold text-slate-800 mb-1">
              Topic-Led Practice
            </div>
            <p className="text-sm text-slate-600">
              Structured lessons with vocabulary building. Great for beginners.
            </p>
          </Link>
        </div>

        <p className="text-xs text-slate-500">
          Both coaches use recasting — silently modeling correct English so you
          learn naturally.
        </p>
      </div>
    </div>
  );
}
