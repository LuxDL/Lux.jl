import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.CA57S8Fv.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56218</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51100</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47104</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44917</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44203</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41052</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37104</span></span>
<span class="line"><span>Validation: Loss 0.37864 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36340 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36508</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35479</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33965</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31443</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30350</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29516</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27122</span></span>
<span class="line"><span>Validation: Loss 0.26573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26242</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24674</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23822</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22438</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21146</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19892</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18720</span></span>
<span class="line"><span>Validation: Loss 0.18470 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17989 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18142</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17023</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16451</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15966</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15239</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14223</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13792</span></span>
<span class="line"><span>Validation: Loss 0.13198 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12884 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12943</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12444</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11834</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11453</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10866</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10455</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09836</span></span>
<span class="line"><span>Validation: Loss 0.09641 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09372 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09569</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09074</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08741</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08478</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07802</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07664</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07094</span></span>
<span class="line"><span>Validation: Loss 0.07161 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06796</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06767</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06428</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06263</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06011</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05785</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05402</span></span>
<span class="line"><span>Validation: Loss 0.05385 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05117 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05351</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04940</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04854</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04597</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04457</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04282</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04116</span></span>
<span class="line"><span>Validation: Loss 0.04077 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03827 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03845</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03855</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03686</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03479</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03572</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03113</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02991</span></span>
<span class="line"><span>Validation: Loss 0.03131 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02902 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03071</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02838</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02877</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02832</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02556</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02435</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02268</span></span>
<span class="line"><span>Validation: Loss 0.02470 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02267 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02324</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02279</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02225</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02237</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02068</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02261</span></span>
<span class="line"><span>Validation: Loss 0.02015 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01837 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01923</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01899</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01883</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01703</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01661</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01706</span></span>
<span class="line"><span>Validation: Loss 0.01693 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01537 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01370</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01448</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01418</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01463</span></span>
<span class="line"><span>Validation: Loss 0.01458 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01324 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01446</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01297</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01284</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01326</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01286</span></span>
<span class="line"><span>Validation: Loss 0.01285 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01165 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01178</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01196</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01209</span></span>
<span class="line"><span>Validation: Loss 0.01149 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01040 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01048</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01070</span></span>
<span class="line"><span>Validation: Loss 0.01039 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00940 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01057</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00915</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00834</span></span>
<span class="line"><span>Validation: Loss 0.00947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00927</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00874</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00759</span></span>
<span class="line"><span>Validation: Loss 0.00871 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00785 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00729</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00750</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00713</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00774</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00766</span></span>
<span class="line"><span>Validation: Loss 0.00748 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00672 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00651</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00683</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00714</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00627 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00658</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00611</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00621</span></span>
<span class="line"><span>Validation: Loss 0.00652 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00585 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00637</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00598</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00514</span></span>
<span class="line"><span>Validation: Loss 0.00611 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00549 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00514</span></span>
<span class="line"><span>Validation: Loss 0.00575 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00516 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00517</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00522</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00508</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00485</span></span>
<span class="line"><span>Validation: Loss 0.00543 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00487 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56054</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50835</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48336</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44937</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42548</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41987</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37171</span></span>
<span class="line"><span>Validation: Loss 0.38888 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36398 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36552</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35909</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33126</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32524</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30358</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28890</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27634</span></span>
<span class="line"><span>Validation: Loss 0.27310 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25683 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25725</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25059</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23813</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22192</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21093</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20523</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19332</span></span>
<span class="line"><span>Validation: Loss 0.18948 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18073 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18229</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17089</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16713</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16151</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15344</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14262</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13437</span></span>
<span class="line"><span>Validation: Loss 0.13525 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12975 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13013</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12527</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12098</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11482</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11128</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10255</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10343</span></span>
<span class="line"><span>Validation: Loss 0.09940 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09457 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09732</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09123</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08793</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08392</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08141</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07773</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06874</span></span>
<span class="line"><span>Validation: Loss 0.07435 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06971 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07195</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06334</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06138</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06115</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05756</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05383</span></span>
<span class="line"><span>Validation: Loss 0.05640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05183 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05419</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05019</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04892</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04828</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04452</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04237</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04192</span></span>
<span class="line"><span>Validation: Loss 0.04304 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03879 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04281</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03996</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03720</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03515</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03279</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03068</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02964</span></span>
<span class="line"><span>Validation: Loss 0.03322 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03079</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03029</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02826</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02749</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02490</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02680</span></span>
<span class="line"><span>Validation: Loss 0.02634 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02296 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02403</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02289</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02254</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02133</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02124</span></span>
<span class="line"><span>Validation: Loss 0.02149 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02001</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01964</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01727</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01686</span></span>
<span class="line"><span>Validation: Loss 0.01805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01554 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01506</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01606</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01366</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01352</span></span>
<span class="line"><span>Validation: Loss 0.01556 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01337 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01372</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01419</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01309</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01229</span></span>
<span class="line"><span>Validation: Loss 0.01371 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01176 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01256</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01210</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01134</span></span>
<span class="line"><span>Validation: Loss 0.01226 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01050 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01022</span></span>
<span class="line"><span>Validation: Loss 0.01109 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00948 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01011</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00984</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00841</span></span>
<span class="line"><span>Validation: Loss 0.01013 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00864 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00952</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00877</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00665</span></span>
<span class="line"><span>Validation: Loss 0.00930 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00732</span></span>
<span class="line"><span>Validation: Loss 0.00861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00736</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00670</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00678 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00732</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00682</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00699</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00644</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00677</span></span>
<span class="line"><span>Validation: Loss 0.00745 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00632 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00647</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00623</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00614</span></span>
<span class="line"><span>Validation: Loss 0.00697 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00659</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00589</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00572</span></span>
<span class="line"><span>Validation: Loss 0.00655 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00554 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00550</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00593</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00434</span></span>
<span class="line"><span>Validation: Loss 0.00616 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00521 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00545</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00504</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00579</span></span>
<span class="line"><span>Validation: Loss 0.00581 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00491 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA_Driver_jll: 0.9.1+1</span></span>
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
