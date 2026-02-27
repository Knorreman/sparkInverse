ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.18"

val sparkVersion = "4.1.1"
// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.15" % Test
// https://mvnrepository.com/artifact/dev.ludovic.netlib/lapack
libraryDependencies += "dev.ludovic.netlib" % "lapack" % "3.0.3"

lazy val root = (project in file("."))
  .settings(
    name := "sparkInverse",
    javaOptions ++= Seq(
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED"
    ),
    Test / javaOptions += "-Xmx8g",
    run / javaOptions ++= Seq(
      "-Xmx8g",
      "-Dspark.driver.memory=8g",
      "-Dspark.executor.memory=8g"
    ),
    Test / fork := true,
    run / fork := true
  )
