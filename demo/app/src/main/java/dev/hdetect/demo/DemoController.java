package dev.hdetect.demo;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.DescribeClusterOptions;
import org.springframework.kafka.core.KafkaAdmin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api")
public class DemoController {
    private final QuestionProducer producer;
    private final VerdictRelay relay;
    private final KafkaAdmin admin;
    private final DemoProperties props;

    public DemoController(QuestionProducer producer, VerdictRelay relay, KafkaAdmin admin, DemoProperties props) {
        this.producer = producer;
        this.relay = relay;
        this.admin = admin;
        this.props = props;
    }

    @GetMapping("/verdicts/stream")
    public SseEmitter stream() {
        return relay.subscribe();
    }

    @PostMapping("/stream/start")
    public Map<String, Object> start(@RequestParam(defaultValue = "0") int offset,
                                     @RequestParam(defaultValue = "0") int limit) throws IOException {
        boolean started = producer.start(offset, limit);
        return Map.of("started", started, "total", producer.total());
    }

    @PostMapping("/stream/stop")
    public Map<String, Object> stop() {
        producer.stop();
        return Map.of("stopped", true);
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        boolean connected;
        try (AdminClient client = AdminClient.create(admin.getConfigurationProperties())) {
            client.describeCluster(new DescribeClusterOptions().timeoutMs(1500)).clusterId().get(2, TimeUnit.SECONDS);
            connected = true;
        } catch (Exception e) {
            connected = false;
        }
        return Map.of("kafka_connected", connected, "streaming", producer.running(), "sent", producer.sent(),
                "total", producer.total(), "judge_cost_usd", props.judgeCostUsd());
    }
}
