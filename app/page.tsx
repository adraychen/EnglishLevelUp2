import Link from "next/link";
import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/auth";

export default async function Home() {
  const user = await getCurrentUser();

  // Redirect logged-in users to dashboard
  if (user) {
    redirect("/dashboard");
  }

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
          <div className="block p-6 bg-white border border-slate-200 rounded-xl shadow-sm">
            <div className="text-2xl mb-2">Dora</div>
            <div className="text-lg font-semibold text-slate-800 mb-1">
              Casual Conversation
            </div>
            <p className="text-sm text-slate-600">
              Free, natural chat on any subject. Great for intermediate to
              advanced learners.
            </p>
          </div>

          <div className="block p-6 bg-white border border-slate-200 rounded-xl shadow-sm">
            <div className="text-2xl mb-2">Morgan</div>
            <div className="text-lg font-semibold text-slate-800 mb-1">
              Topic-Led Practice
            </div>
            <p className="text-sm text-slate-600">
              Structured lessons with vocabulary building. Great for beginners.
            </p>
          </div>
        </div>

        <div className="space-y-3">
          <Link
            href="/login"
            className="block w-full py-3 bg-blue-600 text-white font-medium rounded-xl hover:bg-blue-700 transition"
          >
            Sign In
          </Link>
          <Link
            href="/register"
            className="block w-full py-3 border border-slate-300 text-slate-700 font-medium rounded-xl hover:bg-slate-50 transition"
          >
            Create Account
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
