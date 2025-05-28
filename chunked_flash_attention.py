from keras import layers, ops


class Attention(layers.Layer):
    def __init__(self, output_dim, q_chunk_size=16, kv_chunk_size=1, **kwargs):
        super().__init__(**kwargs)
        self.to_q = layers.Dense(output_dim, use_bias=True, )
        self.to_k = layers.Dense(output_dim, use_bias=True, )
        self.to_v = layers.Dense(output_dim, use_bias=True, )
        self.out_proj = layers.Dense(output_dim, use_bias=True, )
        self.scale = output_dim ** -0.5
        self.output_dim = output_dim
        self.q_chunk_size = q_chunk_size
        self.kv_chunk_size = kv_chunk_size

    @staticmethod
    def _flash_attention(query, key, value, scale, q_chunk_size=1, kv_chunk_size=1):
        result = []
        if kv_chunk_size == 1:
            q_chunks = ops.split(query, q_chunk_size, axis=1)
            for q_chunk_idx in range(q_chunk_size):
                q = q_chunks[q_chunk_idx]
                score = ops.einsum('bqh,bkh->bqk', q, key) * scale
                context = ops.softmax(score)
                acc = ops.einsum('bqk,bkh->bqh', context, value)
                result.append(acc)
        else:
            q_chunks = ops.split(query, q_chunk_size, axis=1)
            k_chunks = ops.split(key, kv_chunk_size, axis=1)
            v_chunks = ops.split(value, kv_chunk_size, axis=1)
            for q_chunk_idx in range(q_chunk_size):
                global_max = 0.0 - float("inf")
                global_max_diffs = 0.0
                acc = 0.0
                q = q_chunks[q_chunk_idx]
                for kv_chunk_idx in range(kv_chunk_size):
                    k = k_chunks[kv_chunk_idx]
                    v = v_chunks[kv_chunk_idx]
                    attn_weights = ops.einsum('bqh,bkh->bqk', q, k) * scale
                    curr_max_score = ops.maximum(ops.max(attn_weights, axis=-1, keepdims=True), global_max)
                    global_max_diffs = global_max_diffs * ops.exp(global_max - curr_max_score)
                    exp_weights = ops.exp(attn_weights - curr_max_score)
                    curr_max_diffs = ops.sum(exp_weights, axis=-1, keepdims=True) + global_max_diffs
                    acc = (acc * global_max_diffs + ops.einsum('bqk,bkh->bqh', exp_weights, v)) / curr_max_diffs
                    global_max_diffs = curr_max_diffs
                    global_max = curr_max_score
                result.append(acc)
        return ops.concatenate(result, axis=1)

    def call(self, x):
        batch_size = ops.shape(x)[0]
        shape = ops.shape(x)
        h, w, c = shape[1], shape[2], shape[3]
        x = ops.reshape(x, (-1, h * w, c))  # b, hw, c
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        out_shape = (batch_size, h, w, self.output_dim)
        if self.q_chunk_size == 1 and self.kv_chunk_size == 1:
            score = ops.einsum('bqh,bkh->bqk', q, k) * self.scale
            context = ops.softmax(score)
            attn = ops.einsum('bqk,bkh->bqh', context, v)
        else:
            attn = self._flash_attention(q, k, v, self.scale, q_chunk_size=self.q_chunk_size,
                                         kv_chunk_size=self.kv_chunk_size)
        out = ops.reshape(attn, out_shape)
        return self.out_proj(out)


class CrossAttention(layers.Layer):
    def __init__(self, num_heads, head_size, q_chunk_size=16, kv_chunk_size=1, **kwargs):
        super().__init__(**kwargs)
        self.to_q = layers.Dense(num_heads * head_size, use_bias=False, name="to_q")
        self.to_k = layers.Dense(num_heads * head_size, use_bias=False, name="to_k")
        self.to_v = layers.Dense(num_heads * head_size, use_bias=False, name="to_v")
        self.scale = head_size ** -0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = layers.Dense(num_heads * head_size, name="to_out")
        self.q_chunk_size = q_chunk_size
        self.kv_chunk_size = kv_chunk_size

    @staticmethod
    def _flash_attention(query, key, value, scale, q_chunk_size=1, kv_chunk_size=1):
        result = []
        if kv_chunk_size == 1:
            q_chunks = ops.split(query, q_chunk_size, axis=1)
            for q_chunk_idx in range(q_chunk_size):
                q = q_chunks[q_chunk_idx]
                score = ops.einsum('bqh,bkh->bqk', q, key) * scale
                context = ops.softmax(score)
                acc = ops.einsum('bqk,bkh->bqh', context, value)
                result.append(acc)
        else:
            q_chunks = ops.split(query, q_chunk_size, axis=1)
            k_chunks = ops.split(key, kv_chunk_size, axis=1)
            v_chunks = ops.split(value, kv_chunk_size, axis=1)
            for q_chunk_idx in range(q_chunk_size):
                global_max = 0.0 - float("inf")
                global_max_diffs = 0.0
                acc = 0.0
                q = q_chunks[q_chunk_idx]
                for kv_chunk_idx in range(kv_chunk_size):
                    k = k_chunks[kv_chunk_idx]
                    v = v_chunks[kv_chunk_idx]
                    attn_weights = ops.einsum('bqh,bkh->bqk', q, k) * scale
                    curr_max_score = ops.maximum(ops.max(attn_weights, axis=-1, keepdims=True), global_max)
                    global_max_diffs = global_max_diffs * ops.exp(global_max - curr_max_score)
                    exp_weights = ops.exp(attn_weights - curr_max_score)
                    curr_max_diffs = ops.sum(exp_weights, axis=-1, keepdims=True) + global_max_diffs
                    acc = (acc * global_max_diffs + ops.einsum('bqk,bkh->bqh', exp_weights, v)) / curr_max_diffs
                    global_max_diffs = curr_max_diffs
                    global_max = curr_max_score
                result.append(acc)
        return ops.concatenate(result, axis=1)

    def call(self, inputs, context=None):
        context = inputs if context is None else context
        batch_size = ops.shape(inputs)[0]
        q_time = ops.shape(inputs)[1]
        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)
        out_shape = (batch_size, q_time, self.num_heads * self.head_size)
        num_heads_axis = -2
        v_out = []
        q_vec = ops.split(q, self.num_heads, axis=-1)
        k_vec = ops.split(k, self.num_heads, axis=-1)
        v_vec = ops.split(v, self.num_heads, axis=-1)
        for idx in range(self.num_heads):
            q = q_vec[idx]
            k = k_vec[idx]
            v = v_vec[idx]
            if self.q_chunk_size == 1 and self.kv_chunk_size == 1:
                score = ops.einsum('bqh,bkh->bqk', q, k) * self.scale
                context = ops.softmax(score)
                attn = ops.einsum('bqk,bkh->bqh', context, v)
            else:
                attn = self._flash_attention(q, k, v, self.scale, q_chunk_size=self.q_chunk_size,
                                             kv_chunk_size=self.kv_chunk_size)
            v_out.append(attn)
        attn = ops.stack(v_out, axis=num_heads_axis)
        out = ops.reshape(attn, out_shape)
        return self.out_proj(out)