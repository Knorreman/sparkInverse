ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.13.18"

val sparkVersion = "4.1.1"
val scalatestVersion = "3.2.19"

lazy val commonSettings = Seq(
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.scalatest" %% "scalatest" % scalatestVersion % Test
  ),
  javaOptions ++= Seq(
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED"
  ),
  Test / javaOptions += "-Xmx8g",
  Test / envVars ++= {
    val threads = sys.props.getOrElse("sparkInverse.openblasThreads", "1")
    Map(
      "OPENBLAS_NUM_THREADS" -> threads,
      "OMP_NUM_THREADS" -> threads,
      "GOTO_NUM_THREADS" -> threads
    )
  },
  Test / fork := true
)

lazy val root = (project in file("."))
  .aggregate(core, bench)
  .settings(
    name := "sparkInverse"
  )

lazy val core = (project in file("core"))
  .settings(commonSettings)
  .settings(
    name := "sparkInverse-core"
  )

lazy val bench = (project in file("bench"))
  .enablePlugins(AssemblyPlugin)
  .dependsOn(core)
  .settings(commonSettings)
  .settings(
    name := "sparkInverse-bench",
    assemblyMergeStrategy := {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case _ => MergeStrategy.first
    },
    run / javaOptions ++= Seq(
      "-Xmx16g",
      "-Dspark.driver.memory=16g",
      "-Dspark.executor.memory=16g"
    ),
    run / envVars ++= {
      val threads = sys.props.getOrElse("sparkInverse.openblasThreads", "1")
      Map(
        "OPENBLAS_NUM_THREADS" -> threads,
        "OMP_NUM_THREADS" -> threads,
        "GOTO_NUM_THREADS" -> threads
      )
    },
    run / fork := true
  )
