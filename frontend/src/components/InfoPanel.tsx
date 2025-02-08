import { useState, useEffect } from "react";
import MarkdownContent from "./MarkdownContent";
import ResearchersList, { Researcher, WorksIn } from "./ResearchersList";
import { Topic } from "../types/Topic";

interface InfoPanelProps {
  topics: Topic[];
  selectedTopic: [number, string];
  loading: boolean;
  error: string | null;
}

const InfoPanel: React.FC<InfoPanelProps> = ({
  topics,
  selectedTopic,
  loading,
  error,
}) => {
  const selectedTopicId = selectedTopic[0];
  const selectedTopicName = selectedTopic[1];

  // When activeSection is null, we show the description.
  // When activeSection === "researchers", we show the researchers view.
  const [activeSection, setActiveSection] = useState<"researchers" | null>(null);

  // Cache dictionary mapping topic ID to its description.
  const [descDict, setDescDict] = useState<Record<number, string>>({});

  // Cache for researchers and works_in data.
  const [researchers, setResearchers] = useState<Researcher[] | null>(null);
  const [worksIn, setWorksIn] = useState<WorksIn[] | null>(null);

  // Build description dictionary from topics.
  useEffect(() => {
    if (topics.length > 0) {
      const dict: Record<number, string> = {};
      topics.forEach((topic) => {
        dict[topic.id] = topic.description;
      });
      setDescDict(dict);
    }
  }, [topics]);

  // Reset the view when the selected topic changes.
  useEffect(() => {
    setActiveSection(null);
  }, [selectedTopicId]);

  // Toggle the researchers view. When turning it on for the first time, fetch researchers & works_in.
  const handleButtonClick = async () => {
    if (activeSection === "researchers") {
      setActiveSection(null);
    } else {
      if (!researchers || !worksIn) {
        try {
          const [resResearchers, resWorksIn] = await Promise.all([
            fetch("http://localhost:3000/researchers").then((r) => r.json()),
            fetch("http://localhost:3000/works_in").then((r) => r.json()),
          ]);
          setResearchers(resResearchers);
          setWorksIn(resWorksIn);
        } catch (err) {
          console.error("Error fetching researchers or works_in:", err);
        }
      }
      setActiveSection("researchers");
    }
  };

  return (
    <div className="info-panel">
      {selectedTopicName === "" && <h3>Select a topic to begin.</h3>}
      {selectedTopicName !== "" && (
        <>
          <div className="info-panel-header">
            <h3>{selectedTopicName}</h3>
            <button
              onClick={handleButtonClick}
              className={activeSection === "researchers" ? "active-button" : ""}
            >
              Researchers
            </button>
          </div>

          <div className="info-panel-content">
            {activeSection === "researchers" ? (
              // Only render ResearchersList when data is available.
              researchers && worksIn ? (
                <ResearchersList
                  topics={topics}
                  selectedTopicName={selectedTopicName}
                  selectedTopicID={selectedTopicId}
                  researchers={researchers}
                  worksIn={worksIn}
                />
              ) : (
                <p>Loading researchers...</p>
              )
            ) : (
              // Default view: show the topic description.
              <>
                {descDict[selectedTopicId] ? (
                  <MarkdownContent markdown={descDict[selectedTopicId]} />
                ) : (
                  <p>No description available!</p>
                )}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default InfoPanel;
