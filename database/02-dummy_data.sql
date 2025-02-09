INSERT INTO topic (id, name, description, parent_id, level, is_leaf) VALUES
(1, 'Computer Science', NULL, NULL, 1, false),
  (2, 'Artificial Intelligence', 'Study of intelligent agents and learning systems.', 1, 2, false),
    (3, 'Machine Learning', 'Enabling computers to learn from data.', 2, 3, false),
      (4, 'Neural Networks', 'Modeling complex patterns using layers of neurons.', 3, 4, true),
      (5, 'Decision Trees', 'Tree-structured models for decision making.', 3, 4, true),
    (6, 'Natural Language Processing', 'Understanding and generating human language.', 2, 3, false),
      (7, 'Sentiment Analysis', 'Determining sentiment in text.', 6, 4, true),
      (8, 'Text Generation', 'AI-based text generation models.', 6, 4, true),
  (9, 'Cybersecurity', 'Protecting digital systems from threats.', 1, 2, false),
    (10, 'Cryptography', 'Securing communication using encryption.', 9, 3, false),
      (11, 'Quantum Cryptography', 'Using quantum mechanics for encryption.', 10, 4, true),
      (12, 'Public Key Infrastructure', 'Managing cryptographic keys.', 10, 4, true);


INSERT INTO paper (arxiv_id, topic_id, title, num_authors) VALUES
('2301.00123', 3, 'A Survey on Machine Learning Techniques', 4),
('2301.00234', 4, 'Advances in Neural Network Architectures', 3),
('2301.00345', 6, 'Sentiment Analysis using Transformer Models', 2),
('2301.00456', 10, 'Quantum Cryptography: A Secure Future', 5),
('2301.00567', 5, 'Decision Tree Optimization Strategies', 3);


INSERT INTO researcher (id, name, link, affiliation) VALUES
(1, 'Dr. Alan Turingbot', 'http://turingbot.ai', 'University of Computron'),
(2, 'Prof. Ada LovelaceGPT', 'http://adalovelacegpt.com', 'Tech University'),
(3, 'Dr. Hash Overflow', 'http://hashoverflow.net', 'CyberSecure Academy'),
(4, 'Prof. Bit Flip', 'http://bitflip.quanta', 'Quantum U'),
(5, 'Dr. Sally Stacktrace', 'http://sallystack.dev', 'Bugfix University');



INSERT INTO writes (researcher_id, arxiv_id, author_position) VALUES
(1, '2301.00123', 1), -- Dr. Turingbot wrote the ML survey
(2, '2301.00234', 1), -- Prof. LovelaceGPT worked on Neural Networks
(3, '2301.00456', 1), -- Dr. Hash Overflow leading in Quantum Crypto
(4, '2301.00456', 2), -- Prof. Bit Flip as second author
(5, '2301.00567', 1); -- Dr. Stacktrace optimizing Decision Trees


INSERT INTO works_in (topic_id, researcher_id, score) VALUES
(3, 1, 92.5), -- Dr. Turingbot specializes in ML
(4, 2, 88.0), -- Prof. LovelaceGPT on Neural Networks
(10, 3, 90.0), -- Dr. Hash Overflow on Cryptography
(11, 4, 95.5), -- Prof. Bit Flip on Quantum Cryptography
(5, 5, 85.0); -- Dr. Stacktrace on Decision Trees

INSERT INTO works_in (topic_id, researcher_id, score) VALUES
(7, 1, 89.0),    -- Dr. Turingbot also on Sentiment Analysis
(8, 2, 91.0),    -- Prof. LovelaceGPT also on Text Generation
(6, 3, 87.5),    -- Dr. Hash Overflow on Natural Language Processing
(12, 4, 93.0),   -- Prof. Bit Flip on Public Key Infrastructure
(2, 5, 88.5),    -- Dr. Sally Stacktrace also on Artificial Intelligence
(3, 2, 84.5),    -- Prof. LovelaceGPT also on Machine Learning
(7, 5, 80.0),    -- Dr. Sally Stacktrace also on Sentiment Analysis
(10, 1, 90.0),   -- Dr. Turingbot also on Cryptography
(8, 3, 85.0),    -- Dr. Hash Overflow also on Text Generation
(11, 5, 88.0);   -- Dr. Sally Stacktrace also on Quantum Cryptography
