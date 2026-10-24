package dev.hdetect.demo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;

/** Streams demo questions onto the questions topic, one at a time with a random delay. */
@Component
public class QuestionProducer {
    private static final Logger log = LoggerFactory.getLogger(QuestionProducer.class);

    private final KafkaTemplate<String, String> kafka;
    private final ObjectMapper json;
    private final DemoProperties props;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicInteger sent = new AtomicInteger();
    private volatile int total;

    public QuestionProducer(KafkaTemplate<String, String> kafka, ObjectMapper json, DemoProperties props) {
        this.kafka = kafka;
        this.json = json;
        this.props = props;
    }

    /** Starts streaming from question `offset` (0-based); at most `limit` questions, 0 for all. */
    public boolean start(int offset, int limit) throws IOException {
        List<String> lines = Files.readAllLines(Path.of(props.questionsFile()));
        int end = limit > 0 ? Math.min(lines.size(), offset + limit) : lines.size();
        List<String> batch = lines.subList(Math.min(offset, end), end);
        if (!running.compareAndSet(false, true)) {
            return false;
        }
        total = batch.size();
        sent.set(0);
        Thread.ofVirtual().name("question-producer").start(() -> stream(batch));
        return true;
    }

    public void stop() {
        running.set(false);
    }

    public boolean running() {
        return running.get();
    }

    public int sent() {
        return sent.get();
    }

    public int total() {
        return total;
    }

    private void stream(List<String> batch) {
        try {
            for (String line : batch) {
                if (!running.get()) {
                    break;
                }
                ObjectNode q = (ObjectNode) json.readTree(line);
                q.put("schema_version", props.schemaVersion());
                q.put("produced_at", Instant.now().truncatedTo(ChronoUnit.MILLIS).toString());
                kafka.send(props.questionsTopic(), q.get("question_id").asText(), json.writeValueAsString(q)).get();
                sent.incrementAndGet();
                Thread.sleep(ThreadLocalRandom.current().nextLong(props.minDelayMs(), props.maxDelayMs() + 1));
            }
        } catch (Exception e) {
            log.error("question stream stopped", e);
        } finally {
            running.set(false);
        }
    }
}
