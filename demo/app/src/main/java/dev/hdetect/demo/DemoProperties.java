package dev.hdetect.demo;

import java.util.Map;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "demo")
public record DemoProperties(
        String schemaVersion,
        String questionsTopic,
        String verdictsTopic,
        String questionsFile,
        long minDelayMs,
        long maxDelayMs,
        Map<String, Double> judgeCostUsd) {
}
