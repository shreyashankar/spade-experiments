"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const fs = __importStar(require("fs"));
const Papa = __importStar(require("papaparse"));
const path = __importStar(require("path"));
function parseCSV(filePath) {
    return new Promise((resolve, reject) => {
        fs.readFile(filePath, "utf8", (err, data) => {
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
                complete: (results) => {
                    resolve(results.data);
                },
            });
        });
    });
}
function parseLabelCSV(filePath) {
    return new Promise((resolve, reject) => {
        fs.readFile(filePath, "utf8", (err, data) => {
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
                complete: (results) => {
                    resolve(results.data);
                },
            });
        });
    });
}
function padLeft(str, length) {
    return str.length < length ? " ".repeat(length - str.length) + str : str;
}
function orderExamples(data) {
    // TODO: handle selected functions. If a selected function
    // fails an example, the example should not show up in the ranking
    // because we know it is a falure.
    const functionStats = {};
    const failureIdScores = {};
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
        const failureRate = functionStats[row.function_name].failures /
            functionStats[row.function_name].total;
        const failureScore = (!row.result ? 1 : 0) * (1 - failureRate);
        const response = row.response;
        failureIdScores[response] = (failureIdScores[response] || 0) + failureScore;
    });
    // Order IDs by failure confidence
    const orderedFailureScores = Object.entries(failureIdScores)
        .map(([response, score]) => ({ response, score }))
        .sort((a, b) => b.score - a.score);
    return orderedFailureScores;
}
function visualizeData(filePath, labelPath) {
    return __awaiter(this, void 0, void 0, function* () {
        const rows = yield parseCSV(filePath);
        const failureScores = orderExamples(rows);
        // Load labelPath
        const csvData = yield parseLabelCSV(labelPath);
        const responseLabelMap = new Map(csvData.map((row) => [row.response, row.label]));
        let outputLines = [];
        // Visualize each score
        failureScores.forEach((idScore, index) => {
            const label = responseLabelMap.get(idScore.response);
            const bar = "=".repeat(Math.round(idScore.score * 10) + 1);
            const indexStr = `#${index + 1}`;
            const paddedIndexStr = padLeft(indexStr, 4);
            const roundedScore = idScore.score.toFixed(2);
            const color = label ? "green" : "red";
            outputLines.push(`${paddedIndexStr} Response: <span style="color: ${color};">${bar}</span> (${roundedScore})<br>`);
        });
        return outputLines.join("\n");
    });
}
function processFolder(basePath, folderName) {
    return __awaiter(this, void 0, void 0, function* () {
        const folderPath = path.join(basePath, folderName);
        const files = fs.readdirSync(folderPath);
        const assertionFile = files.find((file) => file.startsWith("assertion_res_"));
        const labelFile = "labeled_responses.csv";
        if (assertionFile) {
            const assertionFilePath = path.join(folderPath, assertionFile);
            const labelFilePath = path.join(folderPath, labelFile);
            const output = yield visualizeData(assertionFilePath, labelFilePath);
            return `<h2>${folderName}</h2>\n${output}`;
        }
        return "";
    });
}
function processAllFolders(basePath) {
    return __awaiter(this, void 0, void 0, function* () {
        const folders = fs
            .readdirSync(basePath, { withFileTypes: true })
            .filter((dirent) => dirent.isDirectory())
            .map((dirent) => dirent.name);
        // Get rid of only the subsumption folder
        folders.splice(folders.indexOf("subsumption"), 1);
        let allOutputs = "<html><body><pre>";
        for (const folder of folders) {
            const folderOutput = yield processFolder(basePath, folder);
            allOutputs += folderOutput;
        }
        allOutputs += "</pre></body></html>";
        return allOutputs;
    });
}
// Example usage
const basePath = "/Users/shreyashankar/Documents/projects/spade-experiments/paper_experiments";
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
