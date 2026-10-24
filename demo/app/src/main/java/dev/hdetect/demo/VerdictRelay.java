package dev.hdetect.demo;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.concurrent.CopyOnWriteArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/** Consumes verdict events and pushes them to every connected browser over server-sent events. */
@Component
public class VerdictRelay {
    private static final Logger log = LoggerFactory.getLogger(VerdictRelay.class);

    private final ObjectMapper json;
    private final DemoProperties props;
    private final CopyOnWriteArrayList<SseEmitter> emitters = new CopyOnWriteArrayList<>();

    public VerdictRelay(ObjectMapper json, DemoProperties props) {
        this.json = json;
        this.props = props;
    }

    public SseEmitter subscribe() {
        SseEmitter emitter = new SseEmitter(0L);
        emitters.add(emitter);
        emitter.onCompletion(() -> emitters.remove(emitter));
        emitter.onTimeout(() -> emitters.remove(emitter));
        emitter.onError(e -> emitters.remove(emitter));
        return emitter;
    }

    @KafkaListener(topics = "${demo.verdicts-topic}")
    public void onVerdict(String value) throws IOException {
        ObjectNode event = (ObjectNode) json.readTree(value);
        if (!props.schemaVersion().equals(event.path("schema_version").asText())) {
            log.warn("skipping verdict with schema {}", event.path("schema_version").asText());
            return;
        }
        ((ObjectNode) event.with("timestamps")).put("relayed", Instant.now().truncatedTo(ChronoUnit.MILLIS).toString());
        String payload = json.writeValueAsString(event);
        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(SseEmitter.event().name("verdict").data(payload));
            } catch (IOException | IllegalStateException e) {
                emitters.remove(emitter);
            }
        }
    }
}
