"use client"

import React, { useState } from "react";
import Image from "next/image";
import { DefaultApi } from "@services/api"; // Import the DefaultApi class
import { Button } from "@/components/ui/button";
import { TrendingUp } from "lucide-react";
import { Bar } from "recharts";
import { BarChart } from "recharts";
import { XAxis } from "recharts";
import { YAxis } from "recharts";
import { Input } from "@/components/ui/input"

import { Textarea } from "@/components/ui/textarea";
import FlipCard from "./flip";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { type ChartConfig, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import NeoVisGraph from "./db-viewer"; // Adjust the import path as needed

const chartData = [
    { browser: "chrome", visitors: 275, fill: "var(--color-chrome)" },
    { browser: "safari", visitors: 200, fill: "var(--color-safari)" },
    { browser: "firefox", visitors: 187, fill: "var(--color-firefox)" },
    { browser: "edge", visitors: 173, fill: "var(--color-edge)" },
    { browser: "other", visitors: 90, fill: "var(--color-other)" },
]

const chartConfig = {
    visitors: {
        label: "Visitors",
    },
    chrome: {
        label: "Chrome",
        color: "hsl(var(--chart-1))",
    },
    safari: {
        label: "Safari",
        color: "hsl(var(--chart-2))",
    },
    firefox: {
        label: "Firefox",
        color: "hsl(var(--chart-3))",
    },
    edge: {
        label: "Edge",
        color: "hsl(var(--chart-4))",
    },
    other: {
        label: "Other",
        color: "hsl(var(--chart-5))",
    },
} satisfies ChartConfig

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // const [response, setResponse] = useState<{person: string, suspicion_score: number, reasoning: string}[]>([]);
  const [response, setResponse] = useState<any[]>([]);

  const apiInstance = new DefaultApi(); // Create an instance of DefaultApi

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    try {
    const response = await apiInstance.processVideoProcessVideoPost(selectedFile);
    const formattedResponse = response.data.map((item: any) => item.query);
      // .filter((item: any) => typeof item.person === 'string' && typeof item.suspicion_score === 'number' && typeof item.reasoning === 'string')
      // .map((item: any) => ({
      //   person: item.person,
      //   suspicion_score: item.suspicion_score,
      //   reasoning: item.reasoning,
      // }));
    setResponse(formattedResponse);

    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
          {/*<NeoVisGraph></NeoVisGraph>*/}
          <div className="flex gap-4 items-center flex-col sm:flex-row">
          <Input type="file" accept="video/*" onChange={handleFileChange} />
          <Button
            className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            onClick={handleUpload}
          >
            Upload Evidence Footage
          </Button>
        </div>
          <div className="flip-container">
          </div>
          <Card>
              <CardHeader>
                  <CardTitle>Bar Chart - Mixed</CardTitle>
                  <CardDescription>January - June 2024</CardDescription>
              </CardHeader>
              <CardContent>
                  <div className="response-container">
                      {response.length > 0 && (
                          <ul>
                              {response.map((item, index) => (
                                  <li key={index}>
                                      <strong>Person:</strong> {item.person}, <strong>Suspicion
                                      Score:</strong> {item.suspicion_score}, <strong>Reasoning:</strong> {item.reasoning}
                                  </li>
                              ))}
                          </ul>
                      )}
                  </div>

                  {/*<ChartContainer config={chartConfig}>*/}
                  {/*    <BarChart*/}
                  {/*        accessibilityLayer*/}
                  {/*        data={chartData}*/}
                  {/*        layout="vertical"*/}
                  {/*        margin={{*/}
                  {/*            left: 0,*/}
                  {/*        }}*/}
                  {/*    >*/}
                  {/*        <YAxis*/}
                  {/*            dataKey="browser"*/}
                  {/*            type="category"*/}
                  {/*            tickLine={false}*/}
                  {/*            tickMargin={10}*/}
                  {/*            axisLine={false}*/}
                  {/*            tickFormatter={(value) => chartConfig[value as keyof typeof chartConfig]?.label}*/}
                  {/*        />*/}
                  {/*        <XAxis dataKey="visitors" type="number" hide />*/}
                  {/*        <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />*/}
                  {/*        <Bar dataKey="visitors" layout="vertical" radius={5} />*/}
                  {/*    </BarChart>*/}
                  {/*</ChartContainer>*/}
              </CardContent>
              <CardFooter className="flex-col items-start gap-2 text-sm">
              </CardFooter>
          </Card>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Learn
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Examples
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to nextjs.org →
        </a>
      </footer>
    </div>

  );

}
