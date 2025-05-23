<?php
// server.php
// PHP server script with UUID whitelist authentication and line tracking by index for correctness

// Configuration
define('DATA_FILE', '_netstruct.txt');
define('RESULTS_FILE', 'model_results.txt');
// Whitelist of authorized UUIDs; add authorized device UUIDs here
$UUID_WHITELIST = [
    'uuid', //Konrad
];

// Validate UUID format (basic)
function isValidUUID($uuid) {
    return preg_match('/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$/', $uuid);
}

// Authenticate client UUID against whitelist
function authenticate($whitelist) {
    if (!isset($_REQUEST['key']) || !isValidUUID($_REQUEST['key'])) {
        http_response_code(401);
        echo json_encode(['error' => 'Unauthorized: Invalid or missing UUID key']);
        exit;
    }
    $key = $_REQUEST['key'];
    if (!in_array($key, $whitelist)) {
        http_response_code(403);
        echo json_encode(['error' => 'Forbidden: UUID not authorized']);
        exit;
    }
    return $key;
}

// On GET: find first line starting with '-' and mark it 'p', return line content and index
function getAndMarkLineAsProcessing() {
    if (!file_exists(DATA_FILE)) {
        http_response_code(500);
        echo json_encode(['error' => 'Data file not found']);
        exit;
    }
    $lines = file(DATA_FILE, FILE_IGNORE_NEW_LINES);
    if ($lines === false) {
        http_response_code(500);
        echo json_encode(['error' => 'Unable to read data file']);
        exit;
    }
    foreach ($lines as $index => $line) {
        $trimmedLine = ltrim($line);
        if (strpos($trimmedLine, '-') === 0) {
            // Extract content without leading "-"
            $content = ltrim(substr($trimmedLine, 1));
            // Replace leading '-' with 'p', preserving spaces
            $pos = strpos($line, '-');
            $lines[$index] = substr($line, 0, $pos) . 'p' . substr($line, $pos + 1);
            if (file_put_contents(DATA_FILE, implode(PHP_EOL, $lines) . PHP_EOL) === false) {
                http_response_code(500);
                echo json_encode(['error' => 'Failed to update data file']);
                exit;
            }
            return ['line' => $content, 'index' => $index];
        }
    }
    return null;
}

// On POST: mark the line at given index starting with 'p' as completed '#' and save result and line content
function markLineCompleteByIndex($lineIndex, $clientUUID, $result, $model, $epoch) {
    if (!file_exists(DATA_FILE)) {
        return false;
    }
    $lines = file(DATA_FILE, FILE_IGNORE_NEW_LINES);
    if ($lines === false || !isset($lines[$lineIndex])) {
        return false;
    }
    $line = $lines[$lineIndex];
    $trimmedLine = ltrim($line);
    if (strpos($trimmedLine, 'p') !== 0) {
        return false;
    }
    // Prepare line content without leading 'p' and leading spaces preserved
    $pos = strpos($line, 'p');
    $originalLineContent = substr($line, 0, $pos) . substr($line, $pos + 1);

    // Replace leading 'p' with '#'
    $lines[$lineIndex] = substr($line, 0, $pos) . '#' . substr($line, $pos + 1);

    if (file_put_contents(DATA_FILE, implode(PHP_EOL, $lines) . PHP_EOL) === false) {
        return false;
    }

    // Write both the original line content and the result to results file for traceability
    $toWrite = $clientUUID . ' : RESULT: ' . $result . ' : EPOCH: ' . $epoch . ' : MODEL STRUCT: ' . $model;
    if (file_put_contents(RESULTS_FILE, $toWrite . PHP_EOL, FILE_APPEND | LOCK_EX) === false) {
        return false;
    }

    return true;
}

// Main handler
header('Content-Type: application/json');
$clientUUID = authenticate($UUID_WHITELIST);
$method = $_SERVER['REQUEST_METHOD'];

if ($method === 'GET') {
    $result = getAndMarkLineAsProcessing();
    if ($result === null) {
        echo json_encode(['message' => 'No line starting with "-" found']);
    } else {
        echo json_encode(['line' => $result['line'], 'line_index' => $result['index']]);
    }
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Get the query parameter 'key' from the URL
    $key = isset($_GET['key']) ? $_GET['key'] : null;

    // Read the JSON payload from the request body
    $input = json_decode(file_get_contents('php://input'), true);

    // Check for required fields in the JSON payload
    if (!isset($input['result']) || !isset($input['line_index']) || !isset($input['model']) || !isset($input['epoch'])) {
        http_response_code(400);
        echo json_encode(['error' => 'Missing result, model, epoch or line_index field']);
        exit;
    }

    $result = trim($input['result']);
    $lineIndex = (int)$input['line_index'];
    $model = trim($input['model']);
    $epoch = (int)$input['epoch'];


    // Check if the key matches the authenticated client UUID
    if ($key !== $clientUUID) {
        http_response_code(401);
        echo json_encode(['error' => 'Key does not match authenticated client']);
        exit;
    }

    // Check if found index is negative - generate error if so
    if ($foundIndex < 0) {
        http_response_code(400);
        echo json_encode(['error' => 'Found line index is invalid (negative)']);
        exit;
    }

    // Mark line complete and write result & original line to results
    if (!markLineCompleteByIndex($lineIndex, $clientUUID, $result, $model, $epoch)) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to mark line as complete or write result']);
        exit;
    }

    echo json_encode(['message' => 'Result saved, line marked complete']);
    exit;
}

http_response_code(405);
echo json_encode(['error' => 'Method not allowed']);
exit;
?>