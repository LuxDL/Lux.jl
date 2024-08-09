import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.BvIA7MGi.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR833/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR833/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR833/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR833/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56104</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50731</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46854</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45316</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43163</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39840</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38982</span></span>
<span class="line"><span>Validation: Loss 0.38790 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38426 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36493</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35213</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32921</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31958</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30098</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28365</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27078</span></span>
<span class="line"><span>Validation: Loss 0.27009 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26800 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25902</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25086</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23107</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21955</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20355</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20067</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.17864</span></span>
<span class="line"><span>Validation: Loss 0.18639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18537 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17518</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17394</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15822</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15592</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15057</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14370</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.12490</span></span>
<span class="line"><span>Validation: Loss 0.13351 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13274 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12803</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12215</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11769</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11052</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10493</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10511</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09395</span></span>
<span class="line"><span>Validation: Loss 0.09837 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09763 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09295</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08963</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08435</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08378</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07783</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07550</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07064</span></span>
<span class="line"><span>Validation: Loss 0.07377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07306 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07211</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06591</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06263</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05916</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05843</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05628</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05334</span></span>
<span class="line"><span>Validation: Loss 0.05586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05520 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05160</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04927</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04513</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04526</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04520</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04379</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04147</span></span>
<span class="line"><span>Validation: Loss 0.04263 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04204 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03968</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03561</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03772</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03447</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03318</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03263</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02673</span></span>
<span class="line"><span>Validation: Loss 0.03295 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03243 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02916</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02943</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02843</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02659</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02653</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02366</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02330</span></span>
<span class="line"><span>Validation: Loss 0.02618 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02572 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02538</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02280</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02118</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02062</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01969</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01823</span></span>
<span class="line"><span>Validation: Loss 0.02142 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01937</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01913</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01811</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01650</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01480</span></span>
<span class="line"><span>Validation: Loss 0.01805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01772 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01728</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01645</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01440</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01358</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01395</span></span>
<span class="line"><span>Validation: Loss 0.01560 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01531 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01308</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01153</span></span>
<span class="line"><span>Validation: Loss 0.01375 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01349 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01063</span></span>
<span class="line"><span>Validation: Loss 0.01230 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01208 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01018</span></span>
<span class="line"><span>Validation: Loss 0.01113 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01093 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00816</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00996 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00876</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00775</span></span>
<span class="line"><span>Validation: Loss 0.00933 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00917 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00736</span></span>
<span class="line"><span>Validation: Loss 0.00863 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00685</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00781</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00661</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Validation: Loss 0.00747 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00733 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00675</span></span>
<span class="line"><span>Validation: Loss 0.00698 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00588</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00617</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00594</span></span>
<span class="line"><span>Validation: Loss 0.00655 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00503</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00505</span></span>
<span class="line"><span>Validation: Loss 0.00615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00604 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00542</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00561</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00465</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00494</span></span>
<span class="line"><span>Validation: Loss 0.00581 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00569 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56265</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50921</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49510</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45331</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41638</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40908</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39768</span></span>
<span class="line"><span>Validation: Loss 0.36473 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36218 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36035</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35592</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34161</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31992</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31662</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28615</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27307</span></span>
<span class="line"><span>Validation: Loss 0.25730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26115</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24900</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23178</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22422</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21893</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19948</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19636</span></span>
<span class="line"><span>Validation: Loss 0.18175 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18122 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18309</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17401</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16700</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15922</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15159</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14505</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13964</span></span>
<span class="line"><span>Validation: Loss 0.13134 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13106 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13239</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12542</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12097</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11553</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11115</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10641</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10151</span></span>
<span class="line"><span>Validation: Loss 0.09612 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09755</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09145</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08884</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08649</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08156</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07881</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07554</span></span>
<span class="line"><span>Validation: Loss 0.07091 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07043 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07162</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06957</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06560</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06312</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06123</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05972</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05666</span></span>
<span class="line"><span>Validation: Loss 0.05268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05213 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05518</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05090</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04990</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04755</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04510</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04449</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04321</span></span>
<span class="line"><span>Validation: Loss 0.03936 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03882 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04033</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03903</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03798</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03568</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03445</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03362</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03466</span></span>
<span class="line"><span>Validation: Loss 0.02981 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02932 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03143</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02874</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02964</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02947</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02599</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02577</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02430</span></span>
<span class="line"><span>Validation: Loss 0.02327 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02283 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02486</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02488</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02292</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02198</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02168</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01927</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02127</span></span>
<span class="line"><span>Validation: Loss 0.01883 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01909</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01854</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01945</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01819</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01642</span></span>
<span class="line"><span>Validation: Loss 0.01577 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01543 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01628</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01526</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01438</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01422</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01405</span></span>
<span class="line"><span>Validation: Loss 0.01357 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01327 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01233</span></span>
<span class="line"><span>Validation: Loss 0.01194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01167 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01219</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01087</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01106</span></span>
<span class="line"><span>Validation: Loss 0.01066 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01042 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01087</span></span>
<span class="line"><span>Validation: Loss 0.00963 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00903</span></span>
<span class="line"><span>Validation: Loss 0.00877 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00857 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00982</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00786 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00809</span></span>
<span class="line"><span>Validation: Loss 0.00742 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00724 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00801</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00728</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00770</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Validation: Loss 0.00688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00671 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00695</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00675</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00644</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00743</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00625 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00700</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00675</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00650</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00591</span></span>
<span class="line"><span>Validation: Loss 0.00598 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00584 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Validation: Loss 0.00561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00547 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00567</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00559</span></span>
<span class="line"><span>Validation: Loss 0.00527 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00514 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00508</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00456</span></span>
<span class="line"><span>Validation: Loss 0.00497 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00485 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
