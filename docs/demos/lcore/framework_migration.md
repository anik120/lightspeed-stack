### Migrating LCORE from llama-stack to An LLM Orchestration framework 
(eg Langchain/Langraph/Pydantic etc)

Anik Bhattacharjee,
anbhatta@redhat.com

---

## The problem

* We know we have to move away from llama-stack as the backend
* Lot of High Priority work being blocked by this migration (eg Agent Skills)
* Freezing feature development for extended period of time to do the migration is not practical

---

## The Solution

**Feature Flag**

* How all software projects progress work without affecting main line production
* Introduce "`use_orchestration_framework`" feature flag
* Build out all endpoints using new orchestration framework behind feature flag
* Deprecate llama-stack as the backend immediately

---

## The Solution

**Feature Flag**

```
diff --git a/lightspeed-stack.yaml b/lightspeed-stack.yaml
--- a/lightspeed-stack.yaml
+++ b/lightspeed-stack.yaml
@@ -1,21 +1,46 @@
-llama_stack:
-  # Uses a remote llama-stack service
-  # The instance would have already been started with a llama-stack-run.yaml file
-  use_as_library_client: false
-  # Alternative for "as library use"
-  # use_as_library_client: true
-  # library_client_config_path: <path-to-llama-stack-run.yaml-file>
-  url: http://llama-stack:8321
-  api_key: xyzzy
+
+# Feature flags for gradual orchestration framework introduction
+feature_flags:
+  use_orchestration_framework: true
+  endpoints_with_framework: []  # Empty = all endpoints use orch framework 
+  when "`use_orchestration_framework`"=true
+
+# Orchestration framework configuration
+langchain:
+  providers:
+    openai:
+      # API key will be read from OPENAI_API_KEY environment variable
+      # Or set explicitly: api_key: "sk-..."
+      models:
+        - gpt-4
+        - gpt-4-turbo
+        - gpt-3.5-turbo
+      timeout: 60
+      max_retries: 3
+  default_provider: openai
+  default_model: gpt-4
+  enable_streaming: true
+  enable_tracing: false
```

---
See some of the [migration commits in action](https://github.com/anik120/lightspeed-stack/commits/langchain-migration/)

### Work items

* <span style="color: green;">Add feature flag infra for dual-mode backend support</span>
* <span style="color: green;">Migrate /models endpoint to dual-mode architecture</span>
* <span style="color: green;">Migrate /shields endpoint to dual-mode architecture</span>

---

### Work items

* <span style="color: green;">Migrate /providers endpoint to dual-mode architecture</span>
* <span style="color: green;">Migrate /info endpoint to dual-mode architecture</span>
* ...(15 total endpoints) 

<p style="color: red; font-size: 1.5em; font-weight: bold;">
Note: llama-stack is still the default backend at this point
</p>

Deliver Agent Skills/other high priority features with new Orchestration framework that were previously blocked because of llama-stack

---

### Work items

(Assuming Orchestration framework = Langchain from here onwards for ease of communication..)

* <span style="color: green;">Migrate /mcp-servers to dual-mode architecture</span>
	- <span style="color: red;">Add tool calling support</span>
* <span style="color: green;">Migrate /tools to dual-mode architecture</span>
	- MCP tool discovery and conversion to LangChain format
	- Tool calling loop (invoke → tool execution → reinvoke)
	- Human-in-the-loop
	...

---

### Work items

* <span style="color: green;">Migrate /query (non-streaming) to dual-mode architecture</span>
	- * <span style="color: green;">Add LangChain provider registry and LLM providers</span>
	- * <span style="color: green;">Add LangChain provider registry for OpenAI/Azure APIs</span> 
	- Database/Storage:                                          
		- Conversation history retrieval from cache
		- Turn ID generation
		...
	- Conversation Compaction
	...
---

### Work items

* Remaining "Core Langchain infrastruture" build out 
	- RAG Integration with Langchain retrivers
		1. BYOK RAG integration
		2. Vector store integration
		3...
	...
---

### Work items

* Remaining "Core Langchain infrastruture" build out 
	- Streaming query endpoint with SSE
		1. SSE event formatting for LangChain streaming
		2. Streaming with tool calls support
		3...
	...

---

## What that means for Q2

* [LCORE-836] Streamline lightspeed-stack config 
	- Introduce feature flag in config and feature flag infra as part of this EPIC. 
* [LCORE-1284] Provide a low-overhead deployment option for server mode
	- We won't need this anymore since all of llama-stack config will be deprecated
* [LCORE-1349] Supporting backport changes for releases
	- Already in progress (possible already done?)

---

## What that means for Q2

* [LCORE-1339] Support for Agent Skills
	- We need minimum langchain infrastrature in place to deliver this behind feature flag
	- ~ 50 points of work items from "Introduce New Orchestration Framework" only

* [LCORE-260] Support AWS Bedrock
	- This is just ~50 lines of code adding an LLM provider to the existing list of providers
	- https://github.com/anik120/lightspeed-stack/tree/langchain-migration/src/langchain_providers	

---

## What that means for Q2

* These are just langchain configuration (~10-50 LOC) not "reinvent-the-wheel" research projects anymore
	* [LCORE-1311] Conversation History Summarization (Compaction)
	* [LCORE-268] Support Human-in-the-Loop 
* Spend remaining time in Q2 building out langchain migration

---

## What that means for Q2

* Push these out to Q3 
	* Complete remaining work in "Langchain migration"
	* [LCORE-1340] Support for interactive UI elements (MCP Apps)
	* [LCORE-1471] Additional content support in BYOK
	* [LCORE-268] Support Human-in-the-Loop 
		- HITL needs much of the Langchain infrastrure to be built out so most likely push to Q4 in reality
---

### Communication to our customers
### (Red Hat's Lightspeed product suite)

* Write a Brief and send it out to the community to let them know we're moving to an orchestration framework. **In progress...** - https://docs.google.com/document/d/1tXNKI9uSI2ntxBRGVOTnbAgd_xIJcZjPKaG-avfEQ_E/edit?usp=sharing
* Let them know next release of LCORE will have "`use_orchestration_framework`" flag - and all new features will be delivered behind this feature flag
* Engage OLS, and assess how this move affects their "LCORE onboarding" timeline

---

### Communication to our customers

* Next LCORE release will also have the llama-stack backend deprecated
* This will give RHEL/Ansible Lightspeed 2 releases to plan to deal with side effects of our migration
* After two LCORE releases, we will remove EVERYTHING related to llama-stack (as the backend) - we will only keep llama-stack as an API provider going forward

---

* (live demo)
