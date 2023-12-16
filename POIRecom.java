package com.seniorProject.project;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.*;


public class POIRecom {
    private static final int K = 3;  // Number of top-k vectors
    private static PriorityQueue<Pair> topKQueue = new PriorityQueue<>(K);

    public static void main(String[] args) throws IOException {
        // Load user history
        String[] userHistory = {
                "34.4266787,-119.7111968,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5",
                "10,10,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5"
        };

        // Process business data from JSON file
        processBusinessData("yelp_business.json", userHistory);
    }

    private static void processBusinessData(String filePath, String[] userHistory) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                JSONObject jsonObject = parseJsonString(line);
                if (jsonObject != null) {
                    processBusiness(jsonObject, userHistory);
                }
            }
        }
    }

    private static void processBusiness(JSONObject jsonObject, String[] userHistory) {
        try {
            double latitude = Double.parseDouble(jsonObject.get("latitude").toString());
            double longitude = Double.parseDouble(jsonObject.get("longitude").toString());
            String categories = jsonObject.get("categories").toString();
            String stars = jsonObject.get("stars").toString();
            String businessId = jsonObject.get("business_id").toString();
            String oneHotEncoding = getOneHotEncoding(categories);

            // Calculate similarity for each user's history
            for (String history : userHistory) {
                double[] targetVector = convertToDoubleArray(history);
                double[] vector = convertToDoubleArray(businessId + "," + latitude + "," + longitude + "," + oneHotEncoding + "," + stars);
                double similarity = calculateCosSimilarity(targetVector, vector);

                // Output top-k recommendations
                addToQueue(businessId, businessId + "," + latitude + "," + longitude + "," + oneHotEncoding + "," + stars, similarity);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void addToQueue(String id, String vector, double similarity) {
        // 向队列中添加向量，维护前k个最大的相似度
        topKQueue.offer(new Pair(id, vector, similarity));

        if (topKQueue.size() > K) {
            topKQueue.poll();  // 移除队列中最小的元素
        }
    }

    private static String getOneHotEncoding(String categories) {
        // Implement one-hot encoding for relevant categories
        String[] categoryArray = categories.split(", ");
        boolean[] encoding = new boolean[25];

        Arrays.fill(encoding, false);

        for (String category : categoryArray) {
            switch (category) {
                case "Food":
                    encoding[0] = true;
                    break;
                case "Italian":
                    encoding[1] = true;
                    break;
                case "Shopping":
                    encoding[2] = true;
                    break;
                case "Pets":
                    encoding[3] = true;
                    break;
                case "Printing Services":
                    encoding[4] = true;
                    break;
                case "Local Services":
                    encoding[5] = true;
                    break;
                case "Electronics":
                    encoding[6] = true;
                    break;
                case "Furniture Stores":
                    encoding[7] = true;
                    break;
                case "Restaurants":
                    encoding[8] = true;
                    break;
                case "Bubble Tea":
                    encoding[9] = true;
                    break;
                case "Bakeries":
                    encoding[10] = true;
                    break;
                case "Fast Food":
                    encoding[11] = true;
                    break;
                case "Sports Wear":
                    encoding[12] = true;
                    break;
                case "Religious Organizations":
                    encoding[13] = true;
                    break;
                case "Fashion":
                    encoding[14] = true;
                    break;
                case "Breakfast & Brunch":
                    encoding[15] = true;
                    break;
                case "Dentists":
                    encoding[16] = true;
                    break;
                case "Health & Medical":
                    encoding[17] = true;
                    break;
                case "Japanese":
                    encoding[18] = true;
                    break;
                case "Automotive":
                    encoding[19] = true;
                    break;
                case "Hotels & Travel":
                    encoding[20] = true;
                    break;
                case "Korean":
                    encoding[21] = true;
                    break;
                case "Bookstores":
                    encoding[22] = true;
                    break;
                case "Bars":
                    encoding[23] = true;
                    break;
                case "IT Services & Computer Repair":
                    encoding[24] = true;
                    break;
                // Add more categories as needed
            }
        }

        StringBuilder oneHotEncoding = new StringBuilder();
        for (boolean value : encoding) {
            oneHotEncoding.append(value ? "1," : "0,");
        }

        return oneHotEncoding.toString().replaceAll(",$", "");  // Remove trailing comma
    }

    private static double[] convertToDoubleArray(String input) {
        // 将逗号分隔的字符串转换为double数组
        String[] stringArray = input.split(",");
        double[] doubleArray = new double[stringArray.length];

        for (int i = 0; i < stringArray.length; i++) {
            doubleArray[i] = Double.parseDouble(stringArray[i]);
        }

        return doubleArray;
    }

    private static double calculateCosSimilarity(double[] vector1, double[] vector2) {
        // 检查零向量
        if (isZeroVector(vector1) || isZeroVector(vector2)) {
            return 0.0; // 可以根据实际需求返回其他值
        }
        // 计算向量的点积
        double dotProduct = dotProduct(vector1, vector2);

        // 计算向量的模长
        double norm1 = vectorNorm(vector1);
        double norm2 = vectorNorm(vector2);

        // 计算cosine similarity(negative)
        return dotProduct / (norm1 * norm2);
    }

    private static boolean isZeroVector(double[] vector) {
        for (double value : vector) {
            if (value != 0.0) {
                return false;
            }
        }
        return true;
    }

    private static double dotProduct(double[] vector1, double[] vector2) {
        double result = 0.0;
        for (int i = 0; i < vector1.length; i++) {
            result += vector1[i] * vector2[i];
        }
        return result;
    }

    private static double vectorNorm(double[] vector) {
        double sum = 0.0;
        for (double v : vector) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    private static class Pair implements Comparable<Pair> {
        String id;
        String vector;
        double similarity;

        public Pair(String id, String vector, double similarity) {
            this.id = id;
            this.vector = vector;
            this.similarity = similarity;
        }

        @Override
        public int compareTo(Pair other) {
            // 按相似度降序排序
            return -Double.compare(other.similarity, this.similarity);
        }
    }
}
