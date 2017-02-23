/**
  * Created by Administrator on 2017/1/19.
  */

// $example on$
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS,LogisticRegressionWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{DenseVector, Vectors}

import scala.collection.mutable.ArrayBuffer



object LR_test {

    def main(args: Array[String]): Unit = {

        //val conf = new SparkConf().setAppName("LogisticRegression").setMaster("local[*]").set("spark.executor.memory", "2g").set("spark.driver.memory", "2g") // local version
        val conf = new SparkConf().setAppName("LogisticRegression").set("spark.executor.memory", "2g").set("spark.driver.memory", "2g") // HDFS version
        val sc = new SparkContext(conf)
        // $example on$

        // Load and parse the data file.
        //val path = "data/train.d"  // local version
        val path = args(0)    // data file's location
        val dataSet = sc.textFile(path).map(x=>x.trim.split(" "))   // text format
        val data = dataSet.map{ x=>
            val y = x.slice(1,x.length).map(_.toDouble)
            LabeledPoint(x(0).toDouble , Vectors.dense(y))
        }
        //val data = MLUtils.loadLibSVMFile(sc, path)  // read LIBSVM format
      //val time = 1
      val time = args(1).toInt

                val precise = new Array[Double](time)
                val NE_sum = new Array[Double](time)
                for (i <- 0 until time) {
                    val splits2 = data.randomSplit(Array(0.7, 0.3))
                    val train2 = splits2(0).cache()
                    val test2 = splits2(1)

                    // Run training algorithm to build the model
                    /*val model = new LogisticRegressionWithLBFGS()
                      .setNumClasses(2)
                      .run(train2)
                      .setThreshold(0.01)*/
                // LR with SGD
                  val numIterations = args(2).toInt
                  val stepSize = args(3).toDouble
                  val miniBatchFraction = 1.0
                  val model2 = LogisticRegressionWithSGD
                    .train(train2, numIterations, stepSize, miniBatchFraction)


                    // Compute raw scores on the test set.
                  model2.clearThreshold() // to get the predicted value for each class, for calculating log loss
                    val ValueAndLabels = test2.map { case LabeledPoint(label, features) =>
                      val value = model2.predict(features)
                      (value, label)
                    }

                    val predictionAndLabels = ValueAndLabels.map(
                      x =>
                        if ( x._1 < 0.5) (0.0,x._2) else (1.0,x._2)
                    )

                    val NE = -ValueAndLabels.map { case (p, v) => (1 + v) / 2 * math.log(p + 0.00000001) + (1 - v) / 2 * math.log(1 - p + 0.00000001) }.mean()
                    //print("training Normalized Cross Entropy = " + NE)
                    // Prediction Accuracy
                    val precision = new MulticlassMetrics(predictionAndLabels).precision
                    //println("Precision = " + precision)
                    NE_sum(i) = NE
                    precise(i) = precision
                }
                println("Normalized Entropy")
                NE_sum.map(x => println(x))
                println(NE_sum.sum / NE_sum.length)
                println("Prediction Accuracy")
                precise.map(x => println(x))
                println(precise.sum / precise.length)
                sc.stop()
        }

}
// scalastyle:on println
