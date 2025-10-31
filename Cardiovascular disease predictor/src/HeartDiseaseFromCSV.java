import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;
import javax.swing.*;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class HeartDiseaseFromCSV {

    public static void main(String[] args) throws Exception {
        // --- Load CSV data ---
        String filePath = "C:/Users/Aa/Desktop/cardio_train.csv"; // CSV path

        ArrayList<Attribute> attrs = new ArrayList<>();
        attrs.add(new Attribute("age"));
        attrs.add(new Attribute("gender"));
        attrs.add(new Attribute("height"));
        attrs.add(new Attribute("weight"));
        attrs.add(new Attribute("ap_hi"));
        attrs.add(new Attribute("ap_lo"));
        attrs.add(new Attribute("cholesterol"));
        attrs.add(new Attribute("gluc"));
        attrs.add(new Attribute("smoke"));
        attrs.add(new Attribute("alco"));
        attrs.add(new Attribute("active"));
        attrs.add(new Attribute("cardio", Arrays.asList("yes", "no")));

        Instances data = new Instances("heart", attrs, 0);
        data.setClassIndex(data.numAttributes() - 1);

        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line = br.readLine(); // skip header
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(";");
            double[] vals = new double[data.numAttributes()];
            vals[0] = Double.parseDouble(parts[1]);
            vals[1] = Double.parseDouble(parts[2]);
            vals[2] = Double.parseDouble(parts[3]);
            vals[3] = Double.parseDouble(parts[4]);
            vals[4] = Double.parseDouble(parts[5]);
            vals[5] = Double.parseDouble(parts[6]);
            vals[6] = Double.parseDouble(parts[7]);
            vals[7] = Double.parseDouble(parts[8]);
            vals[8] = Double.parseDouble(parts[9]);
            vals[9] = Double.parseDouble(parts[10]);
            vals[10] = Double.parseDouble(parts[11]);
            vals[11] = parts[12].equals("1") ? 0 : 1;
            data.add(new DenseInstance(1.0, vals));
        }
        br.close();

        RandomForest rf = new RandomForest();
        rf.buildClassifier(data);

        Evaluation ev = new Evaluation(data);
        ev.crossValidateModel(rf, data, 5, new Random(1));
        System.out.println("Model Accuracy: " + (1 - ev.errorRate()) * 100 + "%");

        // --- GUI Setup ---
        JFrame frame = new JFrame("Cardiovascular Disease Predictor");
        frame.setSize(600, 550);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().setBackground(Color.PINK);
        frame.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(8, 8, 8, 8);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        Font labelFont = new Font("Arial", Font.BOLD, 18);
        Font inputFont = new Font("Arial", Font.PLAIN, 16);

        // --- Create Labels and TextFields ---
        JLabel[] labels = {
                new JLabel("Age:"), new JLabel("Gender (1=male,2=female):"), new JLabel("Height (cm):"),
                new JLabel("Weight (kg):"), new JLabel("Systolic BP (ap_hi):"), new JLabel("Diastolic BP (ap_lo):"),
                new JLabel("Cholesterol (1=normal,2=above,3=high):"), new JLabel("Glucose (1=normal,2=above,3=high):"),
                new JLabel("Smoke (0=no,1=yes):"), new JLabel("Alcohol (0=no,1=yes):"), new JLabel("Active (0=no,1=yes):")
        };
        JTextField[] inputs = new JTextField[labels.length];

        for (int i = 0; i < labels.length; i++) {
            labels[i].setFont(labelFont);
            labels[i].setForeground(Color.RED);
            inputs[i] = new JTextField();
            inputs[i].setFont(inputFont);
            inputs[i].setPreferredSize(new Dimension(250, 35));
            inputs[i].setBorder(BorderFactory.createLineBorder(Color.RED, 2, true));
            gbc.gridx = 0; gbc.gridy = i;
            frame.add(labels[i], gbc);
            gbc.gridx = 1;
            frame.add(inputs[i], gbc);
        }

        JButton predictBtn = new JButton("Predict");
        predictBtn.setFont(new Font("Arial", Font.BOLD, 20));
        JLabel resultLabel = new JLabel("Result: ");
        resultLabel.setFont(new Font("Arial", Font.BOLD, 20));
        resultLabel.setForeground(Color.BLUE);

        gbc.gridx = 0; gbc.gridy = labels.length; gbc.gridwidth = 2;
        frame.add(predictBtn, gbc);
        gbc.gridy = labels.length + 1;
        frame.add(resultLabel, gbc);

        // --- Button Action ---
        predictBtn.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                try {
                    Instance test = new DenseInstance(data.numAttributes());
                    test.setDataset(data);

                    for (int i = 0; i < inputs.length; i++) {
                        double val = Double.parseDouble(inputs[i].getText());
                        test.setValue(i, val);
                    }

                    double pred = rf.classifyInstance(test);
                    String result = data.classAttribute().value((int) pred);
                    if (result.equals("yes")) {
                        resultLabel.setText("Prediction: This person is likely to have Cardiovascular disease.");
                    } else {
                        resultLabel.setText("Prediction: This person is unlikely to have Cardiovascular disease.");
                    }
                } catch (Exception ex) {
                    resultLabel.setText("Invalid input! Please enter numbers correctly.");
                }
            }
        });

        frame.setVisible(true);
    }
}
