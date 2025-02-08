// MarkdownContent.tsx
import React from "react";
import ReactMarkdown from "react-markdown";

interface MarkdownContentProps {
  markdown: string;
}

const MarkdownContent: React.FC<MarkdownContentProps> = ({ markdown }) => {
  return (
    <div className="markdown-content">
      <ReactMarkdown>{markdown || "No Markdown available."}</ReactMarkdown>
    </div>
  );
};

export default MarkdownContent;
