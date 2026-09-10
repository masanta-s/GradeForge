import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import { Layout } from "./components";
import "./index.css";
import AnswerKeyPage from "./pages/AnswerKeyPage";
import DisputesPage from "./pages/DisputesPage";
import ExamPage from "./pages/ExamPage";
import ExamsPage from "./pages/ExamsPage";
import SettingsPage from "./pages/SettingsPage";
import SheetReviewPage from "./pages/SheetReviewPage";
import StudentsPage from "./pages/StudentsPage";

// Charts (recharts) only load when Analytics is opened.
const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage"));

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <ExamsPage /> },
      {
        path: "/exams/:examId",
        element: <ExamPage />,
        children: [
          { index: true, element: <AnswerKeyPage /> },
          { path: "students", element: <StudentsPage /> },
        ],
      },
      { path: "/exams/:examId/sheets/:sheetId", element: <SheetReviewPage /> },
      { path: "/disputes", element: <DisputesPage /> },
      { path: "/analytics", element: <Suspense fallback={null}><AnalyticsPage /></Suspense> },
      { path: "/settings", element: <SettingsPage /> },
    ],
  },
]);

const queryClient = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } } });

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
