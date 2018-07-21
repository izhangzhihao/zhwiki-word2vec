package com.github.izhangzhihao

import java.io._

import scala.collection.mutable
import scala.collection.parallel.immutable.ParMap


class VecReader(val file: File) {

  def this(filename: String) = this(new File(filename))

  val in = new BufferedReader(new FileReader(file))

  def close() = in.close()

  def readLine() = in.readLine()

  def readLines() = in.lines()
}


/** A Scala port of the word2vec model.  This interface allows the user to access the vector representations
  * output by the word2vec tool, as well as perform some common operations on those vectors.  It does NOT
  * implement the actual continuous bag-of-words and skip-gram architectures for computing the vectors.
  *
  * More information on word2vec can be found here: https://code.google.com/p/word2vec/
  *
  * Example usage:
  * {{{
  * val model = new Word2Vec()
  * model.load("vectors.bin")
  * val results = model.distance(List("france"), N = 10)
  *
  * model.pprint(results)
  * }}}
  */
class Word2Vec {

  private var vocab: ParMap[String, Array[Float]] = _

  private var numWords: Int = _

  private var vecSize: Int = _

  def load(filename: String): Unit = {
    val file = new File(filename)
    if (!file.exists()) {
      throw new FileNotFoundException("Binary vector file not found <" + file.toString + ">")
    }

    val reader = new VecReader(file)

    val info = reader.readLine().split(" ")

    numWords = Integer.parseInt(info(0))
    vecSize = Integer.parseInt(info(1))
    println("File contains " + numWords + " words with vector size " + vecSize)

    //import scala.compat.java8.StreamConverters._
    import scala.collection.JavaConverters._
    //TODO: could be better
    vocab = reader.readLines().iterator().asScala.toStream.par.map(line => {
      val meta: Array[String] = line.split(" ")
      val vectors: Array[Float] = meta.drop(1).map(_.toFloat)
      meta(0) -> vectors
    }).toMap

    println("Loaded " + numWords + " words.")

    reader.close()
  }

  def wordsCount: Int = numWords

  def vectorSize: Int = vecSize

  def contains(word: String): Boolean = {
    vocab.get(word).isDefined
  }

  def vector(word: String): Array[Float] = {
    vocab.getOrElse(word, Array.empty)
  }

  /** Compute the Euclidean distance between two vectors.
    *
    * @param vec1 The first vector.
    * @param vec2 The other vector.
    * @return The Euclidean distance between the two vectors.
    */
  def euclidean(vec1: Array[Float], vec2: Array[Float]): Double = {
    assert(vec1.length == vec2.length, "Uneven vectors!")
    var sum = 0.0
    for (i <- vec1.indices) sum += math.pow(vec1(i) - vec2(i), 2)
    math.sqrt(sum)
  }

  /** Compute the Euclidean distance between the vector representations of the words.
    *
    * @param word1 The first word.
    * @param word2 The other word.
    * @return The Euclidean distance between the vector representations of the words.
    */
  def euclidean(word1: String, word2: String): Double = {
    assert(contains(word1) && contains(word2), "Out of dictionary word! " + word1 + " or " + word2)
    euclidean(vocab(word1), vocab(word2))
  }

  /** Compute the cosine similarity score between two vectors.
    *
    * @param vec1 The first vector.
    * @param vec2 The other vector.
    * @return The cosine similarity score of the two vectors.
    */
  def cosine(vec1: Array[Float], vec2: Array[Float]): Double = {
    assert(vec1.length == vec2.length, "Uneven vectors!")
    var dot, sum1, sum2 = 0.0
    for (i <- vec1.indices) {
      dot += (vec1(i) * vec2(i))
      sum1 += (vec1(i) * vec1(i))
      sum2 += (vec2(i) * vec2(i))
    }
    dot / (math.sqrt(sum1) * math.sqrt(sum2))
  }

  /** Compute the cosine similarity score between the vector representations of the words.
    *
    * @param word1 The first word.
    * @param word2 The other word.
    * @return The cosine similarity score between the vector representations of the words.
    */
  def cosine(word1: String, word2: String): Double = {
    assert(contains(word1) && contains(word2), "Out of dictionary word! " + word1 + " or " + word2)
    cosine(vocab(word1), vocab(word2))
  }

  /** Compute the magnitude of the vector.
    *
    * @param vec The vector.
    * @return The magnitude of the vector.
    */
  def magnitude(vec: Array[Float]): Double = {
    math.sqrt(vec.foldLeft(0.0) { (sum, x) => sum + (x * x) })
  }

  /** Normalize the vector.
    *
    * @param vec The vector.
    * @return A normalized vector.
    */
  def normalize(vec: Array[Float]): Array[Float] = {
    val mag = magnitude(vec).toFloat
    vec.map(_ / mag)
  }

  /** Find the vector representation for the given list of word(s) by aggregating (summing) the
    * vector for each word.
    *
    * @param input The input word(s).
    * @return The sum vector (aggregated from the input vectors).
    */
  def sumVector(input: List[String]): Array[Float] = {
    // Find the vector representation for the input. If multiple words, then aggregate (sum) their vectors.
    input.foreach(w => assert(contains(w), "Out of dictionary word! " + w))
    val vector = new Array[Float](vecSize)
    input.foreach(w => for (j <- vector.indices) vector(j) += vocab(w)(j))
    vector
  }

  /** Find N closest terms in the vocab to the given vector, using only words from the in-set (if defined)
    * and excluding all words from the out-set (if non-empty).  Although you can, it doesn't make much
    * sense to define both in and out sets.
    *
    * @param vector The vector.
    * @param inSet  Set of words to consider. Specify None to use all words in the vocab (default behavior).
    * @param outSet Set of words to exclude (default to empty).
    * @param N      The maximum number of terms to return (default to 40).
    * @return The N closest terms in the vocab to the given vector and their associated cosine similarity scores.
    */
  def nearestNeighbors(vector: Array[Float], inSet: Option[Set[String]] = None,
                       outSet: Set[String] = Set[String](), N: Integer = 40)
  : List[(String, Float)] = {
    // For performance efficiency, we maintain the top/closest terms using a priority queue.
    // Note: We invert the distance here because a priority queue will dequeue the highest priority element,
    //       but we would like it to dequeue the lowest scoring element instead.
    val top = new mutable.PriorityQueue[(String, Float)]()(Ordering.by(-_._2))

    // Iterate over each token in the vocab and compute its cosine score to the input.
    var dist = 0f
    val iterator = if (inSet.isDefined) vocab.filterKeys(k => inSet.get.contains(k)).iterator else vocab.iterator
    iterator.foreach(entry => {
      // Skip tokens in the out set
      if (!outSet.contains(entry._1)) {
        dist = cosine(vector, entry._2).toFloat
        if (top.size < N || top.head._2 < dist) {
          top.enqueue((entry._1, dist))
          if (top.length > N) {
            // If the queue contains over N elements, then dequeue the highest priority element
            // (which will be the element with the lowest cosine score).
            top.dequeue()
          }
        }
      }
    })

    // Return the top N results as a sorted list.
    assert(top.length <= N)
    top.toList.sortWith(_._2 > _._2)
  }

  /** Find the N closest terms in the vocab to the input word(s).
    *
    * @param input The input word(s).
    * @param N     The maximum number of terms to return (default to 40).
    * @return The N closest terms in the vocab to the input word(s) and their associated cosine similarity scores.
    */
  def distance(input: List[String], N: Integer = 40): List[(String, Float)] = {
    // Check for edge cases
    if (input.isEmpty) return List.empty
    input.foreach(w => {
      if (!contains(w)) {
        println("Out of dictionary word! " + w)
        return List.empty
      }
    })

    // Find the vector representation for the input. If multiple words, then aggregate (sum) their vectors.
    val vector = sumVector(input)

    nearestNeighbors(normalize(vector), outSet = input.toSet, N = N)
  }

  /** Find the N closest terms in the vocab to the analogy:
    * - [word1] is to [word2] as [word3] is to ???
    *
    * The algorithm operates as follow:
    * - Find a vector approximation of the missing word = vec([word2]) - vec([word1]) + vec([word3]).
    * - Return words closest to the approximated vector.
    *
    * @param word1 First word in the analogy [word1] is to [word2] as [word3] is to ???.
    * @param word2 Second word in the analogy [word1] is to [word2] as [word3] is to ???
    * @param word3 Third word in the analogy [word1] is to [word2] as [word3] is to ???.
    * @param N     The maximum number of terms to return (default to 40).
    * @return The N closest terms in the vocab to the analogy and their associated cosine similarity scores.
    */
  def analogy(word1: String, word2: String, word3: String, N: Integer = 40): List[(String, Float)] = {
    // Check for edge cases
    if (!contains(word1) || !contains(word2) || !contains(word3)) {
      println("Out of dictionary word! " + Array(word1, word2, word3).mkString(" or "))
      return List.empty
    }

    // Find the vector approximation for the missing analogy.
    val vector = new Array[Float](vecSize)
    for (j <- vector.indices)
      vector(j) = vocab(word2)(j) - vocab(word1)(j) + vocab(word3)(j)

    nearestNeighbors(normalize(vector), outSet = Set(word1, word2, word3), N = N)
  }

  /** Rank a set of words by their respective distance to some central term.
    *
    * @param word The central word.
    * @param set  Set of words to rank.
    * @return Ordered list of words and their associated scores.
    */
  def rank(word: String, set: Set[String]): List[(String, Float)] = {
    // Check for edge cases
    if (set.isEmpty) return List.empty
    (set + word).foreach(w => {
      if (!contains(w)) {
        println("Out of dictionary word! " + w)
        return List.empty
      }
    })

    nearestNeighbors(vocab(word), inSet = Option(set), N = set.size)
  }

  /** Pretty print the list of words and their associated scores.
    *
    * @param words List of (word, score) pairs to be printed.
    */
  def pprint(words: List[(String, Float)]) = {
    println("\n%50s".format("Word") + (" " * 7) + "Cosine distance\n" + ("-" * 72))
    println(words.map(s => "%50s".format(s._1) + (" " * 7) + "%15f".format(s._2)).mkString("\n"))
  }

}


/** ********************************************************************************
  * Demo of the Scala ported word2vec model.
  * ********************************************************************************
  */
object RunWord2Vec {

  /** Demo. */
  def main(args: Array[String]) {
    // Load word2vec model from binary file.
    val model = new Word2Vec()
    model.load("/Users/zhazhang/Downloads/zhwiki/word2vec.bin")

    // distance: Find N closest words
    model.pprint(model.distance(List("france"), N = 10))
    model.pprint(model.distance(List("france", "usa")))
    model.pprint(model.distance(List("france", "usa", "usa")))

    // analogy: "king" is to "queen", as "man" is to ?
    model.pprint(model.analogy("king", "queen", "man", N = 10))

    // rank: Rank a set of words by their respective distance to the central term
    model.pprint(model.rank("摩卡", Set("咖啡", "酒店", "美发")))
  }

}
