
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.conf.Configuration;

import java.io.IOException;
import java.io.StringReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.PriorityQueue;

public class POIRecommendation {
	private static final String HDFS_URI = "hdfs://localhost:8020";
    public static class BusinessMap extends Mapper<LongWritable, Text, Text, Text> {
        private Text outputValue = new Text();
        private Text outputKey = new Text();
        private String[] userHistory = {"34.4266787,-119.7111968,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5",
        		"10,10,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5"};

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            try {
                // Parse JSON object from the input line
                JSONParser parser = new JSONParser();
                JSONObject jsonObject = (JSONObject) parser.parse(new StringReader(value.toString()));

                if (jsonObject.get("latitude") != null && jsonObject.get("longitude") != null && jsonObject.get("categories") != null && jsonObject.get("stars") != null) {
                	// Extract relevant fields
                    double latitude = Double.parseDouble(jsonObject.get("latitude").toString());
                    double longitude = Double.parseDouble(jsonObject.get("longitude").toString());
                    String categories = jsonObject.get("categories").toString();
                    String stars = jsonObject.get("stars").toString();
                    String business_id = jsonObject.get("business_id").toString();

                    // Convert categories to one-hot encoding
                    String oneHotEncoding = getOneHotEncoding(categories);

                    // Emit the stars and vector as output
                    for (String history:userHistory){
                        outputKey.set(history);
                        outputValue.set(business_id + "," + latitude + "," + longitude + "," + oneHotEncoding + "," + stars);
                        context.write(outputKey, outputValue);
                    }
                }
                
            } catch (Exception e) {
                // Handle parsing errors or other exceptions
                e.printStackTrace();
            }
        }
    }

    public static class BusinessReduce extends Reducer<Text, Text, DoubleWritable, Text> {
        private static final int K = 3;  // 你想要的前k个向量的数量
        private PriorityQueue<Pair> topKQueue;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            topKQueue = new PriorityQueue<>(K);
        }

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double[] targetVector = convertToDoubleArray(key.toString());
            for (Text value : values) {
            	String idPart = value.toString().substring(0, value.toString().indexOf(","));
            	String doublePart = value.toString().substring(value.toString().indexOf(",") + 1);
                double[] vector = convertToDoubleArray(doublePart);
                // 向队列中添加向量
                addToQueue(idPart, doublePart, calculateCosSimilarity(targetVector,vector));
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            // 输出前k个向量
            while (!topKQueue.isEmpty()) {
                Pair pair = topKQueue.poll();
                context.write(new DoubleWritable(pair.similarity),new Text(pair.id + ": " + pair.vector));
                
            }
        }

        private void addToQueue(String id, String vector, double similarity) {
            // 向队列中添加向量，维护前k个最大的相似度
            topKQueue.offer(new Pair(id, vector, similarity));

            if (topKQueue.size() > K) {
                topKQueue.poll();  // 移除队列中最小的元素
            }
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

        private double[] convertToDoubleArray(String input) {
            // 将逗号分隔的字符串转换为double数组
            String[] stringArray = input.split(",");
            double[] doubleArray = new double[stringArray.length];

            for (int i = 0; i < stringArray.length; i++) {
                doubleArray[i] = Double.parseDouble(stringArray[i]);
            }

            return doubleArray;
        }

    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", HDFS_URI);
        Job job = Job.getInstance(conf, "poi_recommandation");

        job.setJarByClass(POIRecommendation.class);
        job.setMapperClass(BusinessMap.class);
        job.setReducerClass(BusinessReduce.class);

        job.setOutputKeyClass(DoubleWritable.class);
        job.setOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path("/input/yelp_business.json"));  // Input path for business.json
        FileOutputFormat.setOutputPath(job, new Path("/output"));  // Output path

        System.exit(job.waitForCompletion(true) ? 0 : 1);
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
    public static double[][] sortSimilarity(double[] targetVector, double[][] vectors){
        Arrays.sort(vectors, Comparator.comparingDouble(vector -> - calculateCosSimilarity(targetVector, vector)));
        return vectors;
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
}

