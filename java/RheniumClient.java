
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class RheniumClient {

    private static final String API_URL = "http://localhost:8000";

    public static void main(String[] args) {
        System.out.println("Rhenium OS Java Backend Bridge");
        System.out.println("------------------------------");

        try {
            // 1. Check Health
            if (!checkHealth()) {
                System.err.println("Health check failed. Ensure Python API is running.");
                System.exit(1);
            }

            // 2. Perform Segmentation Request
            String studyUid = "1.2.840.113619.2.55.3.42710457.20251215";
            String seriesUid = "1.2.840.113619.2.55.3.42710457.20251215.1";

            System.out.println("\nSending Segmentation Request...");
            String jsonRequest = String.format(
                    "{" +
                            "  \"study_uid\": \"%s\"," +
                            "  \"series_uid\": \"%s\"," +
                            "  \"task\": \"segmentation\"," +
                            "  \"use_synthetic\": true," +
                            "  \"parameters\": {" +
                            "    \"patient_id\": \"PID-JAVA-001\"," +
                            "    \"patient_name\": \"Java^Test\"" +
                            "  }" +
                            "}",
                    studyUid, seriesUid);

            String response = sendPostRequest("/predict", jsonRequest);
            System.out.println("Response Received:");
            System.out.println(response);

            // 3. Perform Classification Request
            System.out.println("\nSending Classification Request...");
            String classRequest = String.format(
                    "{" +
                            "  \"study_uid\": \"%s\"," +
                            "  \"series_uid\": \"%s\"," +
                            "  \"task\": \"classification\"," +
                            "  \"use_synthetic\": true," +
                            "  \"parameters\": {" +
                            "    \"patient_id\": \"PID-JAVA-002\"" +
                            "  }" +
                            "}",
                    studyUid, seriesUid);

            String classResponse = sendPostRequest("/predict", classRequest);
            System.out.println("Classification Result: " + classResponse);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static boolean checkHealth() throws Exception {
        URL url = new URL(API_URL + "/health");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");

        int code = conn.getResponseCode();
        System.out.println("Health Check Status: " + code);

        if (code == 200) {
            try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()))) {
                String line;
                while ((line = br.readLine()) != null) {
                    System.out.println("Health Info: " + line);
                }
            }
            return true;
        }
        return false;
    }

    private static String sendPostRequest(String endpoint, String jsonInputString) throws Exception {
        URL url = new URL(API_URL + endpoint);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setRequestProperty("Accept", "application/json");
        conn.setDoOutput(true);

        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = jsonInputString.getBytes("utf-8");
            os.write(input, 0, input.length);
        }

        try (BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream(), "utf-8"))) {
            StringBuilder response = new StringBuilder();
            String responseLine = null;
            while ((responseLine = br.readLine()) != null) {
                response.append(responseLine.trim());
            }
            return response.toString();
        }
    }
}
