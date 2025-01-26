"use client";

import React, { useEffect, useRef } from "react";
import NeoVis, {NeovisConfig} from "neovis.js";

const NeoVisGraph = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const config: NeovisConfig = {
      containerId: "viz", // Correct property
      server_url: "neo4j+s://d381ec68.databases.neo4j.io", // Replace with your Neo4j connection URL
      server_user: "neo4j", // Replace with your Neo4j username
      server_password: "5vlCBP1hnzfqNyqeLbSuu6PQ_1pY6b0n_XDpfLRQPoU", // Replace with your Neo4j password
    };

    const viz = new NeoVis(config);
    viz.render();

    return () => {
      viz.clearNetwork(); // Cleanup when the component unmounts
    };
  }, []);

  return <div id="viz" ref={containerRef} style={{ width: "100%", height: "500px" }} />;
};

export default NeoVisGraph;