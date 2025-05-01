import java.io.BufferedReader;
import java.io.InputStreamReader;

public class CommandInjectionVuln {

    public static void main(String[] args) {
        try {
            if (args.length < 1) {
                System.out.println("Usage: java CommandInjectionVuln <hostname>");
                return;
            }

            String host = args[0];  // 👈 No input validation
            String cmd = "ping -c 2 " + host;  // Vulnerable to injection

            System.out.println("Executing: " + cmd);
            Process process = Runtime.getRuntime().exec(cmd);
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()));

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
