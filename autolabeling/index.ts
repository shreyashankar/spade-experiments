import * as fs from "fs";
import * as Papa from "papaparse";
import * as path from "path";

interface Row {
  prompt: string;
  example: string;
  response: string;
  function_name: string;
  result: boolean;
}

interface LabelRow {
  response: string;
  label: boolean;
}

interface FunctionStats {
  total: number;
  failures: number;
}

interface IDScore {
  response: string;
  score: number;
}

function parseCSV(filePath: string): Promise<Row[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(
      filePath,
      "utf8",
      (err: NodeJS.ErrnoException | null, data: string) => {
        if (err) {
          reject(err);
          return;
        }

        Papa.parse(data, {
          header: true,
          transform: (value, field) => {
            if (field === "result") {
              return value.toLowerCase() === "true";
            }
            return value;
          },
          complete: (results: Papa.ParseResult<Row>) => {
            resolve(results.data);
          },
        });
      }
    );
  });
}

function parseLabelCSV(filePath: string): Promise<LabelRow[]> {
  return new Promise((resolve, reject) => {
    fs.readFile(
      filePath,
      "utf8",
      (err: NodeJS.ErrnoException | null, data: string) => {
        if (err) {
          reject(err);
          return;
        }

        Papa.parse(data, {
          header: true,
          transform: (value, field) => {
            if (field === "label") {
              return value.toLowerCase() === "1";
            }
            return value;
          },
          complete: (results: Papa.ParseResult<LabelRow>) => {
            resolve(results.data);
          },
        });
      }
    );
  });
}

function padLeft(str: string, length: number): string {
  return str.length < length ? " ".repeat(length - str.length) + str : str;
}

function orderExamples(data: Row[]) {
  // TODO: handle selected functions. If a selected function
  // fails an example, the example should not show up in the ranking
  // because we know it is a falure.

  const functionStats: Record<string, FunctionStats> = {};
  const failureIdScores: Record<string, number> = {};

  // Calculate failure rates
  data.forEach((row) => {
    if (!functionStats[row.function_name]) {
      functionStats[row.function_name] = { total: 0, failures: 0 };
    }

    functionStats[row.function_name].total++;
    if (!row.result) {
      functionStats[row.function_name].failures++;
    }
  });

  // Compute failure and success scores for each (example, prompt, response) tuple
  data.forEach((row) => {
    const failureRate =
      functionStats[row.function_name].failures /
      functionStats[row.function_name].total;
    const failureScore = (!row.result ? 1 : 0) * (1 - failureRate);
    const response = row.response;
    failureIdScores[response] = (failureIdScores[response] || 0) + failureScore;
  });

  // Order IDs by failure confidence
  const orderedFailureScores: IDScore[] = Object.entries(failureIdScores)
    .map(([response, score]) => ({ response, score }))
    .sort((a, b) => b.score - a.score);

  return orderedFailureScores;
}

async function visualizeData(filePath: string, labelPath: string) {
  const rows = await parseCSV(filePath);
  const failureScores = orderExamples(rows);

  // Load labelPath
  const csvData = await parseLabelCSV(labelPath);
  const responseLabelMap = new Map(
    csvData.map((row) => [row.response, row.label])
  );

  let outputLines: string[] = [];

  // Visualize each score
  failureScores.forEach((idScore, index) => {
    const label = responseLabelMap.get(idScore.response);
    const bar = "=".repeat(Math.round(idScore.score * 10) + 1);

    const indexStr = `#${index + 1}`;
    const paddedIndexStr = padLeft(indexStr, 4);
    const roundedScore = idScore.score.toFixed(2);

    const color = label ? "green" : "red";
    outputLines.push(
      `${paddedIndexStr} Response: <span style="color: ${color};">${bar}</span> (${roundedScore})<br>`
    );
  });

  return outputLines.join("\n");
}

async function processFolder(
  basePath: string,
  folderName: string
): Promise<string> {
  const folderPath = path.join(basePath, folderName);
  const files = fs.readdirSync(folderPath);

  const assertionFile = files.find((file) => file.startsWith("assertion_res_"));
  const labelFile = "labeled_responses.csv";

  if (assertionFile) {
    const assertionFilePath = path.join(folderPath, assertionFile);
    const labelFilePath = path.join(folderPath, labelFile);

    const output = await visualizeData(assertionFilePath, labelFilePath);
    return `<h2>${folderName}</h2>\n${output}`;
  }

  return "";
}

async function processAllFolders(basePath: string) {
  const folders = fs
    .readdirSync(basePath, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  // Get rid of the subsumption folder
  folders.splice(folders.indexOf("subsumption"), 1);

  let allOutputs = "<html><body><pre>";

  for (const folder of folders) {
    const folderOutput = await processFolder(basePath, folder);
    allOutputs += folderOutput;
  }

  allOutputs += "</pre></body></html>";

  return allOutputs;
}

// Example usage
const basePath =
  "/Users/shreyashankar/Documents/projects/spade-experiments/paper_experiments";

processAllFolders(basePath)
  .then((allOutputs) => {
    fs.writeFileSync("combined_output.html", allOutputs);
  })
  .catch((err) => {
    console.error("Error:", err);
  });

// Example usage
// const filePath =
//   "/Users/shreyashankar/Documents/projects/spade-experiments/paper_experiments/codereviews/assertion_res_c26484e8b8cb09a62f72d2fa70de3ef466a5a4381bdabe91d0805ceb3eb2929f.csv";

// const labelPath =
//   "/Users/shreyashankar/Documents/projects/spade-experiments/paper_experiments/codereviews/labeled_responses.csv";
// const experimentName = "codereviews";

// visualizeData(filePath, labelPath, experimentName).then((output) => {
//   // Save output to file
//   fs.writeFileSync("output.html", output);
// });
