import { useState, useEffect } from "react";
import ResearchersList, { Researcher, WorksIn } from "./ResearchersList";
import PapersList, { Paper } from "./PapersList";
import { Topic } from "../types/Topic";
import { BACKEND } from "../const";
import axios from "axios";

interface InfoPanelProps {
  topics: Topic[];
  papers: Paper[];
  selectedTopic: Topic;
  loading: boolean;
  error: string | null;
}

const InfoPanel: React.FC<InfoPanelProps> = ({
  topics,
  papers,
  selectedTopic,
  loading,
  error,
}) => {

  // activeSection can be either "papers" or "researchers".
  const [activeSection, setActiveSection] = useState<"papers" | "researchers">("papers");

  // Cache for researchers and works_in data.
  const [researchers, setResearchers] = useState<Researcher[] | null>(null);
  const [worksIn, setWorksIn] = useState<WorksIn[] | null>(null);

  // Reset the view when the selected topic changes.
  useEffect(() => {
    // Reset to "papers" view when the topic changes.
    setActiveSection("papers");
  }, [selectedTopic]);

  // Handle switching to the Papers view
  const handlePapersClick = async () => {
    setActiveSection("papers");
  }

  // Handle switching to the Researchers view.
  const handleResearchersClick = async () => {
    if (!researchers || !worksIn) {
      try {
        const [resResearchers, resWorksIn] = await Promise.all([
          (await axios.get<Researcher[]>(`${BACKEND}/researchers`)),
          (await axios.get<WorksIn[]>(`${BACKEND}/works_in`)),
        ]);
        setResearchers(resResearchers.data);
        setWorksIn(resWorksIn.data);
      } catch (err) {
        console.error("Error fetching researchers or works_in:", err);
      }
    }
    setActiveSection("researchers");
  };

  // Handle switching to the Papers view.

  return (
    <div className="info-panel">
      {selectedTopic.name === "" && <h3>Select a topic to begin.</h3>}
      {selectedTopic.name !== "" && (
        <>
          <div className="info-panel-header">
            <h3>{selectedTopic.name}</h3>
            <div className="toggle-buttons">
              <button
                onClick={handlePapersClick}
                className={activeSection === "papers" ? "active-button" : ""}
                style={{ marginRight: "10px" }} // Gap added here
              >
                Papers
              </button>
              <button
                onClick={handleResearchersClick}
                className={activeSection === "researchers" ? "active-button" : ""}
              >
                Researchers
              </button>
            </div>
          </div>

          <div className="info-panel-content">
            {activeSection === "researchers" ? (
              // Render ResearchersList when the researchers view is active.
              researchers && worksIn ? (
                <ResearchersList
                  topics={topics}
                  selectedTopic={selectedTopic}
                  researchers={researchers}
                  worksIn={worksIn}
                />
              ) : (
                <p>No researchers found...</p>
              )
            ) : (
              // Render PapersList when the papers view is active.
              papers ? (
                <PapersList
                  selectedTopic={selectedTopic}
                  papers={papers}
                />
              ) : (
                <p>No papers found...</p>
              )
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default InfoPanel;
