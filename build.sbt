ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

val sparkVersion = "3.3.1"
// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.15" % Test
// https://mvnrepository.com/artifact/dev.ludovic.netlib/lapack
libraryDependencies += "dev.ludovic.netlib" % "lapack" % "3.0.3"

lazy val root = (project in file("."))
  .settings(
    name := "sparkInverse"
  )
